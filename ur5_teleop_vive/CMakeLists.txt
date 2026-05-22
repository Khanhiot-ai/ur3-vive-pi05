cmake_minimum_required(VERSION 3.8)
project(ur5_teleop_vive)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

# Generate messages
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Xyzrpy.msg"
  DEPENDENCIES std_msgs
)

# Install Python scripts
install(PROGRAMS
  ur5_teleop_vive/thesis_code/vive_ur5_teleop_params.py
  ur5_teleop_vive/thesis_code/ur_follow_using_class_ros2.py
  ur5_teleop_vive/thesis_code/lately_round.py
  ur5_teleop_vive/thesis_code/calib_manual.py
  ur5_teleop_vive/thesis_code/new_lately.py
  ur5_teleop_vive/thesis_code/frame_as_posestamped_ros2.py
  ur5_teleop_vive/thesis_code/vive_tf_and_joy_ros2.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install URBasic module
install(DIRECTORY ur5_teleop_vive/thesis_code/URBasic
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

# Install config files
install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

# Install mesh files
install(DIRECTORY mesh
  DESTINATION share/${PROJECT_NAME}
)

ament_export_dependencies(rosidl_default_runtime)

ament_package()
