# 시선이 일정 시간 한 곳에 머물렀을 때(dwell) 발행되는 확정 응시점
std_msgs/Header header          # frame_id = "map" (SLAM world)
geometry_msgs/Point point       # 응시점 무게중심 [m]
float32 duration                # 응시 유지 시간 [s]
float32 dispersion              # 응시점 산포(RMS 반경) [m], 작을수록 안정적
float32 confidence              # 0.0 ~ 1.0
