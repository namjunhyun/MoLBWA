# 시선이 일정 시간 한 곳에 머물렀을 때(dwell) 발행되는 확정 응시점
std_msgs/Header header          # frame_id = "map" (SLAM world)
geometry_msgs/Point point       # 응시점 무게중심 [m]
float32 duration                # 응시 유지 시간 [s]
float32 dispersion              # 응시점 산포(RMS 반경) [m], 작을수록 안정적
float32 confidence              # 0.0 ~ 1.0

# 응시 구간 동안의 머리(씬 카메라) 평균 위치. point 와 같은 좌표계.
#
# 이게 있으면 target_resolver 가 시선을 "점"이 아니라 "광선"으로 다룰 수 있다.
# origin -> point 방향의 광선을 테이블 평면과 교차시키면 깊이 오차가 가로 오차로
# 새지 않는다 (직교 투영은 머리 0.45m/거리 0.35m 기준 깊이오차의 0.62배가 샌다).
#
# /head/pose 가 안 들어오면 has_origin=false 로 두고, 소비자는 예전처럼
# 직교 투영으로 물러선다.
geometry_msgs/Point origin
bool has_origin
