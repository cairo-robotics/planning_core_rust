use crate::collision::collision_nn::CollisionNN;
use crate::collision::env_collision::*;
use crate::optimization::tsr::TSR;
use crate::spacetime::robot::Robot;
use crate::utils_rust::file_utils::*;
use crate::utils_rust::sampler::ThreadRobotSampler;
use crate::utils_rust::yaml_utils::*;
use nalgebra::{Point3, UnitQuaternion, Vector3};

use ncollide3d::pipeline::*;
use ncollide3d::query::*;
use ncollide3d::shape::*;
use std::ops::Deref;

pub struct AgentVars {
    pub robot: Robot,
    pub sampler: ThreadRobotSampler,
    pub init_state: Vec<f64>,
    pub xopt: Vec<f64>,
    pub prev_state: Vec<f64>,
    pub prev_state2: Vec<f64>,
    pub prev_state3: Vec<f64>,
    pub goal_positions: Vec<Vector3<f64>>,
    pub goal_quats: Vec<UnitQuaternion<f64>>,
    pub init_ee_positions: Vec<Vector3<f64>>,
    pub init_ee_quats: Vec<UnitQuaternion<f64>>,
    pub position_mode_relative: bool, // if false, will be absolute
    pub rotation_mode_relative: bool, // if false, will be absolute
    pub collision_nn: CollisionNN,
    pub env_collision: RelaxedIKEnvCollision,
    pub objective_mode: String,
    pub keyframe_mean_pose: Vec<Vec<f64>>,
    pub planning_tsr: TSR,
    pub secondary_tsr: TSR,
}

impl AgentVars {
    pub fn from_yaml_path(
        settings_fp: String,
        position_mode_relative: bool,
        rotation_mode_relative: bool,
    ) -> Self {
        let info_file_name = get_info_file_name(settings_fp.clone());
        let path_to_config = get_path_to_config();
        let info_fp = path_to_config + "/info_files/" + info_file_name.as_str();
        let ifp = InfoFileParser::from_yaml_path(info_fp.clone());
        let mut robot = Robot::from_yaml_path(info_fp.clone());
        let num_chains = robot.joint_names.len();
        let sampler = ThreadRobotSampler::new(robot.clone());

        let mut goal_positions: Vec<Vector3<f64>> = Vec::new();
        let mut goal_quats: Vec<UnitQuaternion<f64>> = Vec::new();

        let init_ee_positions = robot.get_ee_positions(ifp.starting_config.as_slice());
        let init_ee_quats = robot.get_ee_quats(ifp.starting_config.as_slice());

        for i in 0..num_chains {
            goal_positions.push(init_ee_positions[i]);
            goal_quats.push(init_ee_quats[i]);
        }

        let collision_nn_path =
            get_path_to_config() + "/collision_nn_rust/" + ifp.collision_nn_file.as_str() + ".yaml";
        let collision_nn = CollisionNN::from_yaml_path(collision_nn_path);

        let env_collision_file = EnvCollisionFileParser::from_yaml_path(settings_fp.clone());
        let frames = robot.get_frames_immutable(&ifp.starting_config.clone());
        let env_collision =
            RelaxedIKEnvCollision::init_collision_world(env_collision_file, &frames);
        let objective_mode = get_objective_mode(settings_fp.clone());
        let planning_tsr = TSR::new_from_poses(
            &vec![0.0f64, 0.0, 0.0, 0.0, 0.0, 0.0],
            &vec![0.0f64, 0.0, 0.0, 0.0, 0.0, 0.0],
            &vec![
                vec![-100.0f64, 100.0],
                vec![-100.0f64, 100.0],
                vec![-100.0f64, 100.0],
                vec![-3.14f64, 3.14],
                vec![-3.14f64, 3.14],
                vec![-3.14f64, 3.14],
            ],
        );
        let secondary_tsr = TSR::new_from_poses(
            &vec![0.0f64, 0.0, 0.0, 0.0, 0.0, 0.0],
            &vec![0.0f64, 0.0, 0.0, 0.0, 0.0, 0.0],
            &vec![
                vec![-100.0f64, 100.0],
                vec![-100.0f64, 100.0],
                vec![-100.0f64, 100.0],
                vec![-3.14f64, 3.14],
                vec![-3.14f64, 3.14],
                vec![-3.14f64, 3.14],
            ],
        );

        AgentVars {
            robot,
            sampler,
            init_state: ifp.starting_config.clone(),
            xopt: ifp.starting_config.clone(),
            prev_state: ifp.starting_config.clone(),
            prev_state2: ifp.starting_config.clone(),
            prev_state3: ifp.starting_config.clone(),
            goal_positions,
            goal_quats,
            init_ee_positions,
            init_ee_quats,
            position_mode_relative,
            rotation_mode_relative,
            collision_nn,
            env_collision,
            objective_mode,
            keyframe_mean_pose: vec![vec![1.0, 1.0, 1.0], vec![0.0, 0.0, 0.0, 1.0]],
            planning_tsr: planning_tsr,
            secondary_tsr: secondary_tsr
        }
    }

    pub fn update_keyframe_mean_pose(&mut self, mean_configuration: Vec<Vec<f64>>) {
        self.keyframe_mean_pose = mean_configuration.clone();
    }

    pub fn update(&mut self, xopt: Vec<f64>) {
        self.prev_state3 = self.prev_state2.clone();
        self.prev_state2 = self.prev_state.clone();
        self.prev_state = self.xopt.clone();
        self.xopt = xopt.clone();
    }

    pub fn update_collision_world(&mut self) -> bool {
        let frames = self.robot.get_frames_immutable(&self.xopt);
        self.env_collision.update_links(&frames);
        for event in self.env_collision.world.proximity_events() {
            let c1 = self
                .env_collision
                .world
                .objects
                .get(event.collider1)
                .unwrap();
            let c2 = self
                .env_collision
                .world
                .objects
                .get(event.collider2)
                .unwrap();
            if event.new_status == Proximity::Intersecting {
                println!(
                    "===== {:?} Intersecting of {:?} =====",
                    c1.data().name,
                    c2.data().name
                );
            } else if event.new_status == Proximity::WithinMargin {
                println!(
                    "===== {:?} WithinMargin of {:?} =====",
                    c1.data().name,
                    c2.data().name
                );
                if c1.data().link_data.is_link {
                    let arm_idx = c1.data().link_data.arm_idx as usize;
                    if self.env_collision.active_pairs[arm_idx].contains_key(&event.collider2) {
                        let links = self.env_collision.active_pairs[arm_idx]
                            .get_mut(&event.collider2)
                            .unwrap();
                        if !links.contains(&event.collider1) {
                            links.push(event.collider1);
                        }
                    } else {
                        let links: Vec<CollisionObjectSlabHandle> = vec![event.collider1];
                        self.env_collision.active_pairs[arm_idx].insert(event.collider2, links);
                    }
                } else if c2.data().link_data.is_link {
                    let arm_idx = c2.data().link_data.arm_idx as usize;
                    if self.env_collision.active_pairs[arm_idx].contains_key(&event.collider1) {
                        let links = self.env_collision.active_pairs[arm_idx]
                            .get_mut(&event.collider1)
                            .unwrap();
                        if !links.contains(&event.collider2) {
                            links.push(event.collider2);
                        }
                    } else {
                        let links: Vec<CollisionObjectSlabHandle> = vec![event.collider2];
                        self.env_collision.active_pairs[arm_idx].insert(event.collider1, links);
                    }
                }
            } else {
                println!(
                    "===== {:?} Disjoint of {:?} =====",
                    c1.data().name,
                    c2.data().name
                );
                if c1.data().link_data.is_link {
                    let arm_idx = c1.data().link_data.arm_idx as usize;
                    if self.env_collision.active_pairs[arm_idx].contains_key(&event.collider2) {
                        let links = self.env_collision.active_pairs[arm_idx]
                            .get_mut(&event.collider2)
                            .unwrap();
                        if links.contains(&event.collider1) {
                            let index = links.iter().position(|x| *x == event.collider1).unwrap();
                            links.remove(index);
                        }
                        if links.len() == 0 {
                            self.env_collision.active_pairs[arm_idx].remove(&event.collider2);
                        }
                    }
                } else if c2.data().link_data.is_link {
                    let arm_idx = c2.data().link_data.arm_idx as usize;
                    if self.env_collision.active_pairs[arm_idx].contains_key(&event.collider1) {
                        let links = self.env_collision.active_pairs[arm_idx]
                            .get_mut(&event.collider1)
                            .unwrap();
                        if links.contains(&event.collider2) {
                            let index = links.iter().position(|x| *x == event.collider2).unwrap();
                            links.remove(index);
                        }
                        if links.len() == 0 {
                            self.env_collision.active_pairs[arm_idx].remove(&event.collider1);
                        }
                    }
                }
            }
            // self.print_active_pairs();
        }

        self.env_collision.world.update();

        let link_radius = self.env_collision.link_radius;
        let penalty_cutoff: f64 = link_radius * 2.0;
        let a = penalty_cutoff.powi(2);
        let filter_cutoff = 3;
        for arm_idx in 0..frames.len() {
            // let mut sum_max: f64 = 0.0;
            let mut active_candidates: Vec<(Option<CollisionObjectSlabHandle>, f64)> = Vec::new();
            for key in self.env_collision.active_pairs[arm_idx].keys() {
                let obstacle = self.env_collision.world.objects.get(*key).unwrap();
                // println!("Obstacle: {:?}", obstacle.data());
                let mut sum: f64 = 0.0;
                let last_elem = frames[arm_idx].0.len() - 1;
                for j in 0..last_elem {
                    let start_pt = Point3::from(frames[arm_idx].0[j]);
                    let end_pt = Point3::from(frames[arm_idx].0[j + 1]);
                    let segment = Segment::new(start_pt, end_pt);
                    let segment_pos = nalgebra::one();
                    let dis = distance(
                        obstacle.position(),
                        obstacle.shape().deref(),
                        &segment_pos,
                        &segment,
                    ) - link_radius;
                    // println!("VARS -> {:?}, Link{}, Distance: {:?}", obstacle.data(), j, dis);
                    if dis > 0.0 {
                        sum += a / (dis + link_radius).powi(2);
                    } else if self.objective_mode != "noECA" {
                        return true;
                    } else {
                        break;
                    }
                }
                active_candidates.push((Some(*key), sum));
            }

            // println!("Number of active obstacles: {}", active_obstacles.len());
            if self.objective_mode != "noECA" {
                if active_candidates.len() > filter_cutoff {
                    active_candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                    let active_obstacles = active_candidates[0..filter_cutoff]
                        .iter()
                        .cloned()
                        .collect();
                    self.env_collision.active_obstacles[arm_idx] = active_obstacles;
                } else {
                    self.env_collision.active_obstacles[arm_idx] = active_candidates;
                }
            }
        }
        return false;
    }

    pub fn print_active_pairs(&self) {
        let frames = self.robot.get_frames_immutable(&self.xopt);
        for i in 0..frames.len() {
            for (key, values) in self.env_collision.active_pairs[i].iter() {
                let collider = self.env_collision.world.objects.get(*key).unwrap();
                for v in values {
                    let link = self.env_collision.world.objects.get(*v).unwrap();
                    println!(
                        "Arm {}, Active pair {:?} and {:?}",
                        i,
                        collider.data().name,
                        link.data().name
                    );
                }
            }
        }
    }
}
