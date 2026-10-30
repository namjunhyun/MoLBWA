# Fixation을 "잡을 물체" 또는 "놓을 자리"로 해석한 결과
#
# 좌표계: header.frame_id 가 이 point 를 표현한 좌표계다.
#   target_resolver 는 입력이 무슨 프레임으로 오든 로봇 base_frame 으로 변환해서
#   내보내므로 보통 "base_link" 가 들어간다. 소비자(task_manager/arm_server)는
#   이 값을 확인하고 자기 기준 프레임과 다르면 변환해야 한다.
std_msgs/Header header
geometry_msgs/Point point       # 최종 목표점. PICK이면 파지점, PLACE면 테이블 위 점
string role                     # "pick" | "place"
string label                    # "cup", "table", "unknown"
float32 height_above_table      # 테이블 평면으로부터의 높이 [m]
bool snapped                    # 물체 중심으로 보정(스냅)되었는지
float32 confidence              # 0.0 ~ 1.0. Fixation 의 신뢰도를 그대로 전달한다.
                                # task_manager 가 min_confidence 미만이면 목표를 거부한다.
