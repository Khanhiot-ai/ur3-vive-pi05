from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    declared_arguments = []
    # UR specific arguments
    declared_arguments.append(
        DeclareLaunchArgument(
            "ur_type",
            default_value="ur3",
            description="Type/series of used UR robot.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "description_package",
            default_value="ur_description",
            description="Description package with robot URDF/XACRO files.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "tf_prefix",
            default_value="",
            description="tf_prefix of the joint names, useful for "
            "multi-robot setup. If changed, also joint names in the controllers' configuration "
            "have to be updated.",
        )
    )

    ur_type = LaunchConfiguration("ur_type")
    description_package = LaunchConfiguration("description_package")
    tf_prefix = LaunchConfiguration("tf_prefix")

    # Paths to config files for xacro
    joint_limit_params = PathJoinSubstitution(
        [FindPackageShare(description_package), "config", ur_type, "joint_limits.yaml"]
    )
    physical_params = PathJoinSubstitution(
        [FindPackageShare(description_package), "config", ur_type, "physical_parameters.yaml"]
    )
    visual_params = PathJoinSubstitution(
        [FindPackageShare(description_package), "config", ur_type, "visual_parameters.yaml"]
    )
    kinematics_params = PathJoinSubstitution(
        [FindPackageShare(description_package), "config", ur_type, "default_kinematics.yaml"]
    )
    
    # We also need these for the ros2_control tag in URDF, even if we don't start the driver
    script_filename = PathJoinSubstitution(
        [FindPackageShare("ur_client_library"), "resources", "external_control.urscript"]
    )
    input_recipe_filename = PathJoinSubstitution(
        [FindPackageShare("ur_robot_driver"), "resources", "rtde_input_recipe.txt"]
    )
    output_recipe_filename = PathJoinSubstitution(
        [FindPackageShare("ur_robot_driver"), "resources", "rtde_output_recipe.txt"]
    )

    robot_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution([FindPackageShare(description_package), "urdf", "ur.urdf.xacro"]),
            " ",
            "name:=", "ur",
            " ",
            "ur_type:=", ur_type,
            " ",
            "script_filename:=", script_filename,
            " ",
            "input_recipe_filename:=", input_recipe_filename,
            " ",
            "output_recipe_filename:=", output_recipe_filename,
            " ",
            "tf_prefix:=", tf_prefix,
            " ",
            "joint_limit_params:=", joint_limit_params,
            " ",
            "physical_params:=", physical_params,
            " ",
            "visual_params:=", visual_params,
            " ",
            "kinematics_params:=", kinematics_params,
            " ",
            "safety_limits:=", "true",
            " ",
            "safety_pos_margin:=", "0.15",
            " ",
            "safety_k_position:=", "20",
        ]
    )
    robot_description = {"robot_description": robot_description_content}

    # Robot State Publisher
    node_robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[robot_description],
    )

    # RViz
    rviz_config_file = PathJoinSubstitution(
        [FindPackageShare(description_package), "rviz", "view_robot.rviz"]
    )
    
    node_rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", rviz_config_file],
    )

    # Note: ur_follow_params.py publishes joint_states, so robot_state_publisher will pick them up.
    
    return LaunchDescription(declared_arguments + [node_robot_state_publisher, node_rviz])
