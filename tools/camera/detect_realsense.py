import pyrealsense2 as rs
ctx = rs.context()
print(f"检测到 {len(ctx.devices)} 个RealSense设备")