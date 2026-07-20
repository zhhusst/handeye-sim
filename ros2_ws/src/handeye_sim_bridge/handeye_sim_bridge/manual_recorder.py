#!/usr/bin/env python3
"""
manual_recorder.py — 手动标定数据采集节点 v4
端点数据直接订阅 /gocator/endpoints (scene_publisher 已算好的几何求交结果)

用法:
  python3 src/handeye_sim_bridge/handeye_sim_bridge/manual_recorder.py -o /workspace/data/xxx.json
"""

import rclpy, sys, os, json, numpy as np
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, JointState
from sensor_msgs_py.point_cloud2 import read_points
from tf2_ros import Buffer, TransformListener
from rclpy.time import Time
from std_msgs.msg import Float64MultiArray


def _rpy_to_matrix(rx, ry, rz):
    rx, ry, rz = np.deg2rad(rx), np.deg2rad(ry), np.deg2rad(rz)
    cx, sx = np.cos(rx), np.sin(rx); cy, sy = np.cos(ry), np.sin(ry); cz, sz = np.cos(rz), np.sin(rz)
    return np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]]) @ \
           np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]]) @ \
           np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])


def ros_tf_to_matrix(t):
    q = t.transform.rotation; x, y, z, w = q.x, q.y, q.z, q.w
    R = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                  [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                  [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
    tv = np.array([t.transform.translation.x, t.transform.translation.y, t.transform.translation.z])
    return R, tv


def get_key_nonblock(fd=0):
    import select as sel
    if sel.select([fd], [], [], 0.1)[0]:
        key = os.read(fd, 1)
        while sel.select([fd], [], [], 0.01)[0]: os.read(fd, 1)
        return key
    return None


class ManualRecorder(Node):
    def __init__(self, output_path):
        super().__init__('manual_recorder')
        self.output_path = output_path
        self.records = []

        self.create_subscription(PointCloud2, '/gocator/profile', self._pc_cb, 1)
        self.create_subscription(JointState, '/joint_states', self._js_cb, 1)
        self.create_subscription(Float64MultiArray, '/gocator/endpoints', self._ep_cb, 1)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.latest_profile = None
        self.latest_joints = None
        self.latest_endpoints = None  # [n, e1x, e1y, e1z, v1, e2x, e2y, e2z, v2]

        self.R_he_gt = np.array([[ 0.06048987,  0.88751593, -0.45678927],
                                 [-0.98526871, -0.02024626, -0.16981060],
                                 [-0.15995789,  0.46033200,  0.87321699]])
        self.t_he_gt = np.array([-0.011579, -0.004621, 0.359284])

        self.get_logger().info('Manual Recorder v4 ready (reads /gocator/endpoints).')
        self.get_logger().info('  r=record  s=save  q=quit  p=preview')
        self.create_timer(0.3, self._keyboard_tick)

    def _pc_cb(self, msg):
        try:
            pts = [list(p) for p in read_points(msg, field_names=('x','y','z'), skip_nans=True)]
            self.latest_profile = np.array(pts, dtype=np.float64) if pts else None
        except: pass

    def _js_cb(self, msg):
        JN = ['J1_joint','J2_joint','J3_joint','J4_joint','J5_joint','J6_joint']
        try:
            self.latest_joints = np.array([msg.position[msg.name.index(j)] for j in JN])
        except: pass

    def _ep_cb(self, msg):
        """端点: /gocator/endpoints → [n, e1x, e1y, e1z, v1, e2x, e2y, e2z, v2]"""
        self.latest_endpoints = list(msg.data)

    def _get_flange_pose(self):
        try:
            tf = self.tf_buffer.lookup_transform('world', 'fanuc_flange', Time())
            return ros_tf_to_matrix(tf)
        except: return None, None

    def _record_current(self):
        R_i, t_i = self._get_flange_pose()
        if R_i is None:
            self.get_logger().error('TF lookup failed — simulation running?'); return

        profile = self.latest_profile
        if profile is None:
            self.get_logger().error('No profile data.'); return

        eps = self.latest_endpoints
        if eps is None or len(eps) < 9:
            self.get_logger().error('No endpoint data from /gocator/endpoints'); return

        # Parse: [n, e1x, e1y, e1z, v1, e2x, e2y, e2z, v2]
        n_ep = int(eps[0])
        valid_e1 = eps[4] > 0.5
        valid_e2 = eps[8] > 0.5
        p_S_e1 = np.array([eps[1], eps[2], eps[3]]) if valid_e1 else None
        p_S_e2 = np.array([eps[5], eps[6], eps[7]]) if valid_e2 else None

        scan_pts_S = [np.array([p[0], 0.0, p[2]]) for p in profile]

        record = {
            'R_i': R_i.tolist(), 't_i': t_i.tolist(),
            'scan_pts_S': [p.tolist() for p in scan_pts_S],
            'valid_e1': valid_e1, 'valid_e2': valid_e2,
        }
        if p_S_e1 is not None: record['p_S_e1'] = p_S_e1.tolist()
        if p_S_e2 is not None: record['p_S_e2'] = p_S_e2.tolist()

        self.records.append(record)
        info = f'Recorded #{len(self.records)}  e1={valid_e1} e2={valid_e2}  n_pts={len(scan_pts_S)}  n_ep={n_ep}'
        if self.latest_joints is not None:
            info += f'  joints={np.round(self.latest_joints,3).tolist()}'
        self.get_logger().info(info)

    def _save(self):
        data = {'poses': self.records,
                'scene': {'R_he_gt': self.R_he_gt.tolist(), 't_he_gt': self.t_he_gt.tolist()}}
        os.makedirs(os.path.dirname(self.output_path) or '.', exist_ok=True)
        with open(self.output_path, 'w') as f: json.dump(data, f, indent=2)
        self.get_logger().info(f'Saved {len(self.records)} records → {self.output_path}')

    def _preview(self):
        if not self.records: self.get_logger().info('No records.'); return
        lines = [f'\n{"="*50}', f'Records: {len(self.records)}']
        for i, r in enumerate(self.records):
            e1 = '\u2713' if r['valid_e1'] else '\u2717'
            e2 = '\u2713' if r['valid_e2'] else '\u2717'
            lines.append(f'  #{i+1}: e1={e1} e2={e2}  n_pts={len(r["scan_pts_S"])}')
        lines.append(f'{"="*50}\n'); print('\n'.join(lines))

    def _keyboard_tick(self):
        key = get_key_nonblock()
        if key is None: return
        key = key.decode('utf-8', errors='replace').lower()
        if key == 'r':   self._record_current()
        elif key == 's': self._save()
        elif key == 'q': self._save(); self.get_logger().info('Quit.'); self.destroy_node(); raise SystemExit
        elif key == 'p': self._preview()


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('-o','--output', default='/workspace/data/manual_test.json')
    args, _ = p.parse_known_args()
    rclpy.init()
    node = ManualRecorder(args.output)
    try: rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit): node._save()
    finally: rclpy.shutdown()

if __name__ == '__main__': main()
