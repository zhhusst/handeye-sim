# Local Gocator SDK runtime subset

This directory contains the minimum local SDK subset required to build and run
`gocator_profile_driver`: the GoSdk and kApi headers plus the Linux x64 shared
libraries. It was copied from the lab's previously working Gocator installation.

The SDK is third-party LMI Technologies software, not part of this repository's
Apache-2.0 source code. Before publishing or redistributing the repository,
verify the applicable LMI SDK licence. A system installation can be used instead
by setting `GOCATOR_SDK_ROOT` at build time.
