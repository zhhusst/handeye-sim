#!/usr/bin/env bash
# ROS/ament setup files legitimately probe optional unset variables, so nounset
# must only be enabled after both environments have been sourced.
set -eo pipefail

source /opt/ros/jazzy/setup.bash
source /workspace/ros2_ws/install/setup.bash
set -u

usage() {
  echo "用法：$0 <场景名> [--duration 秒] [--passive]"
  echo "  场景名示例：static、rx、ry、rz、auto_seed"
  echo "  --duration：自动停止时长，默认 30 秒；0 表示按 Ctrl+C 停止"
  echo "  --passive：仅录制，不重置/锁定/控制检测器；自动种子测试必须使用"
}

scenario=""
duration_s="30"
passive="false"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --duration)
      [[ $# -ge 2 ]] || { usage >&2; exit 2; }
      duration_s="$2"
      shift 2
      ;;
    --passive)
      passive="true"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      if [[ -n "${scenario}" ]]; then
        echo "错误：只能指定一个场景名。" >&2
        usage >&2
        exit 2
      fi
      scenario="$1"
      shift
      ;;
  esac
done

if [[ -z "${scenario}" ]]; then
  echo "错误：必须提供场景名。" >&2
  usage >&2
  exit 2
fi
if [[ ! "${scenario}" =~ ^[A-Za-z0-9_-]+$ ]]; then
  echo "错误：场景名只能包含字母、数字、下划线和连字符。" >&2
  exit 2
fi
if [[ ! "${duration_s}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "错误：--duration 必须是非负秒数。" >&2
  exit 2
fi

run_root="/workspace/data/breakpoint_tracking_runs"
mkdir -p "${run_root}"
run_name="$(date -u +%Y%m%d_%H%M%S)_${scenario}"
output="${run_root}/${run_name}"

if ! ros2 topic list | grep -qx '/gocator/profile'; then
  echo "错误：未发现 /gocator/profile，请先启动真机环境并打开激光。" >&2
  exit 1
fi

echo "准备记录双断点跟踪数据：${output}"
echo "场景：${scenario}；时长：${duration_s} 秒；被动模式：${passive}"
echo "脚本只记录传感器与机器人状态，不会向机器人发送运动命令。"

tracking_start_pid=""
cleanup_done="false"

cleanup_tracking_test() {
  if [[ "${cleanup_done}" == "true" ]]; then
    return
  fi
  cleanup_done="true"
  if [[ -n "${tracking_start_pid}" ]] \
    && kill -0 "${tracking_start_pid}" 2>/dev/null; then
    kill -TERM "${tracking_start_pid}" 2>/dev/null || true
    wait "${tracking_start_pid}" 2>/dev/null || true
  fi
  if [[ "${passive}" == "true" ]]; then
    return
  fi
  timeout 10 ros2 topic pub --once /calibration/detection_control \
    std_msgs/msg/String "{data: SEED_TRACK_STOP}" >/dev/null 2>&1 || true
  # A tracking experiment is an isolated perception test. Restore the finite
  # purple alignment ROI so the next run cannot inherit a stale local guide.
  timeout 10 ros2 service call /profile_endpoint_detector/reset \
    std_srvs/srv/Trigger '{}' >/dev/null 2>&1 || true
}

trap cleanup_tracking_test EXIT

status_before=""
if [[ "${passive}" == "false" ]]; then
  echo "正在重置检测导向并锁定当前两个物理断点……"
  timeout 10 ros2 service call /profile_endpoint_detector/reset \
    std_srvs/srv/Trigger '{}' >/dev/null
  lock_output=""
  for _attempt in {1..15}; do
    sleep 0.2
    lock_output="$(
      timeout 10 ros2 service call /profile_endpoint_detector/lock \
        std_srvs/srv/Trigger '{}' 2>&1 || true
    )"
    if [[ "${lock_output}" == *"success=True"* ]]; then
      break
    fi
  done
  if [[ "${lock_output}" != *"success=True"* ]]; then
    echo "错误：当前双断点没有稳定落入紫色 ROI，无法锁定。" >&2
    echo "${lock_output}" >&2
    timeout 10 ros2 service call /profile_endpoint_detector/status \
      std_srvs/srv/Trigger '{}' || true
    exit 1
  fi
  echo "双断点已锁定。录制开始后再按当前场景缓慢移动机器人。"
fi

status_before="$(
  timeout 10 ros2 service call /profile_endpoint_detector/status \
    std_srvs/srv/Trigger '{}' 2>&1 || true
)"

# Start the isolated perception test only after rosbag has subscribed. Passive
# mode leaves all control to the automatic seed node and merely observes it.
if [[ "${passive}" == "false" ]]; then
  (
    sleep 2
    timeout 10 ros2 topic pub --once /calibration/detection_control \
      std_msgs/msg/String "{data: SEED_TRACK_START}" >/dev/null 2>&1
  ) &
  tracking_start_pid=$!
fi

record_command=(ros2 bag record --output "${output}" \
  --disable-keyboard-controls \
  --max-cache-size 16777216 \
  --custom-data "scenario=${scenario}" "passive=${passive}" \
  --topics \
  /gocator/profile_raw_mm \
  /gocator/profile \
  /gocator/profile_2d \
  /gocator/profile_viz \
  /fanuc/joint_states_raw \
  /joint_states \
  /calibration/flange_pose \
  /calibration/endpoints \
  /calibration/target_surface_points \
  /calibration/detection_guide \
  /calibration/detection_control \
  /calibration/detection_prior \
  /calibration/detection_measured_prior \
  /calibration/seed_motion_state \
  /profile_endpoint_detector/diagnostics \
  /parameter_events \
  /tf \
  /tf_static)

set +e
if [[ "${duration_s}" == "0" || "${duration_s}" == "0.0" ]]; then
  echo "开始录制。按 Ctrl+C 结束。"
  "${record_command[@]}"
else
  echo "开始录制，将在 ${duration_s} 秒后自动结束。"
  # High-resolution raw+metric profiles can leave tens of megabytes in the
  # rosbag double buffer. Give MCAP enough time to flush and write its index;
  # a forced kill before that point can remove the incomplete output entirely.
  timeout --signal=INT --kill-after=60 "${duration_s}" \
    "${record_command[@]}"
fi
record_status=$?
set -e

status_after="$(
  timeout 10 ros2 service call /profile_endpoint_detector/status \
    std_srvs/srv/Trigger '{}' 2>&1 || true
)"
cleanup_tracking_test
trap - EXIT
if [[ ${record_status} -ne 0 \
  && ${record_status} -ne 124 \
  && ${record_status} -ne 130 ]]; then
  echo "录制失败：rosbag 退出状态为 ${record_status}，没有可用数据。" >&2
  exit "${record_status}"
fi
if [[ -d "${output}" ]]; then
  {
    echo "scenario=${scenario}"
    echo "duration_s=${duration_s}"
    echo "passive=${passive}"
    echo "recorded_utc=$(date -u --iso-8601=seconds)"
    echo "status_before=${status_before}"
    echo "status_after=${status_after}"
  } >"${output}/experiment.txt"
  cp /workspace/ros2_ws/src/handeye_sim_bridge/config/calibration.yaml \
    "${output}/calibration.yaml"
  cp /workspace/ros2_ws/src/fanuc_gocator_bridge/config/real_calibration.yaml \
    "${output}/real_calibration.yaml"
  timeout 10 ros2 param dump /profile_endpoint_detector \
    >"${output}/detector_parameters.yaml" 2>/dev/null || true
fi
if [[ ! -f "${output}/metadata.yaml" ]]; then
  echo "录制失败：没有生成 metadata.yaml，不能把该目录当作有效 bag。" >&2
  exit 1
fi
if ! ros2 bag info "${output}" >"${output}/bag_info.txt" 2>&1; then
  echo "录制失败：MCAP 索引不可读，请不要使用该数据包。" >&2
  cat "${output}/bag_info.txt" >&2
  exit 1
fi
echo "记录完成：${output}"
