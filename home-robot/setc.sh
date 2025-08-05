ln -s /aiarena/group/eaigroup/ql/docker/home/clip /root/.cache/
ln -s /aiarena/group/eaigroup/ql/docker/home/torch /root/.cache/
ln -s /aiarena/group/eaigroup/ql/docker/home/huggingface /root/.cache/


cp /aiarena/group/eaigroup/ql/ovmm/collect_file_keep/nav_to_obj_task.py /home-robot/src/third_party/habitat-lab/habitat-lab/habitat/tasks/ovmm/sub_tasks/nav_to_obj_task.py
cp /aiarena/group/eaigroup/ql/ovmm/collect_file_keep/nav_to_obj_sensors.py /home-robot/src/third_party/habitat-lab/habitat-lab/habitat/tasks/ovmm/sub_tasks/nav_to_obj_sensors.py
cp /aiarena/group/eaigroup/ql/ovmm/collect_file_keep/habitat_ovmm_env.py /home-robot/src/home_robot_sim/home_robot_sim/env/habitat_ovmm_env/habitat_ovmm_env.py