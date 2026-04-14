from pymycobot import MyCobotSocket

mc = MyCobotSocket("10.16.120.250", 9000)

# Kirim data sudut
mc.send_angles([0, 0, 0, 0, 0, 0], 50)
