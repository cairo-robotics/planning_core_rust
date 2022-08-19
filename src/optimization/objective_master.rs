use crate::core::vars::AgentVars;
use crate::optimization::objective::*;

pub struct ObjectiveMaster {
    pub objectives: Vec<Box<dyn ObjectiveTrait + Send>>,
    pub num_chains: usize,
    pub weight_priors: Vec<f64>,
    pub lite: bool,
    pub finite_diff_grad: bool,
}

impl ObjectiveMaster {
    pub fn standard_ik(num_chains: usize) -> Self {
        let mut objectives: Vec<Box<dyn ObjectiveTrait + Send>> = Vec::new();
        let mut weight_priors: Vec<f64> = Vec::new();
        for i in 0..num_chains {
            objectives.push(Box::new(MatchEEPosGoals::new(i)));
            weight_priors.push(1.0);
            objectives.push(Box::new(MatchEEQuatGoals::new(i)));
            weight_priors.push(1.0);
        }
        Self {
            objectives,
            num_chains,
            weight_priors,
            lite: true,
            finite_diff_grad: true,
        }
    }

    pub fn tune_weight_priors(&mut self, vars: &AgentVars) {
        let a = 0.05;
        let cap = 0.001;
        for i in 0..self.num_chains {
            let mut score_max = 0.0;
            for (option, score) in &vars.env_collision.active_obstacles[i] {
                if *score > score_max {
                    score_max = *score;
                }
            }
            // match ee quat goal objectives
            let weight_cur = self.weight_priors[3 * i + 1];
            let weight_delta = a / (a + score_max) - weight_cur;
            if weight_delta.abs() < cap {
                self.weight_priors[3 * i + 1] += weight_delta;
            } else {
                self.weight_priors[3 * i + 1] += cap * weight_delta / weight_delta.abs();
            }
        }
    }
    

    pub fn relaxed_ik(num_chains: usize, objective_mode: String) -> Self {
        let mut objectives: Vec<Box<dyn ObjectiveTrait + Send>> = Vec::new();
        let mut weight_priors: Vec<f64> = Vec::new();
        for i in 0..num_chains {
            objectives.push(Box::new(MatchEEPosGoals::new(i)));
            weight_priors.push(5.0);
            objectives.push(Box::new(MatchEEQuatGoals::new(i)));
            if objective_mode == "ECA3" {
                weight_priors.push(0.0);
            } else if objective_mode == "ECAA" {
                weight_priors.push(1.0);
            } else {
                weight_priors.push(1.0);
            }
            objectives.push(Box::new(EnvCollision::new(i)));
            if objective_mode == "noECA" {
                weight_priors.push(0.0);
            } else {
                weight_priors.push(1.0);
            }
        }
        objectives.push(Box::new(MinimizeVelocity));
        weight_priors.push(7.0);
        objectives.push(Box::new(MinimizeAcceleration));
        weight_priors.push(2.0);
        objectives.push(Box::new(MinimizeJerk));
        weight_priors.push(1.0);
        objectives.push(Box::new(JointLimits));
        weight_priors.push(1.0);
        objectives.push(Box::new(NNSelfCollision));
        weight_priors.push(1.0);

        Self {
            objectives,
            num_chains,
            weight_priors,
            lite: false,
            finite_diff_grad: true,
        } // fix this
    }

    pub fn omega_optimize(num_chains: usize, objective_mode: String) -> Self {
        let mut objectives: Vec<Box<dyn ObjectiveTrait + Send>> = Vec::new();
        let mut weight_priors: Vec<f64> = Vec::new();
        for i in 0..num_chains {
            objectives.push(Box::new(TSRPosGoal::new(i)));
            weight_priors.push(7.0);
            objectives.push(Box::new(TSRQuatGoal::new(i)));
            if objective_mode == "noECA" {
                weight_priors.push(0.0);
            } else {
                weight_priors.push(5.0);
            }
            // objectives.push(Box::new(PlanningTSRError::new(i)));
            // weight_priors.push(10.0);
            // objectives.push(Box::new(MatchEEPosGoals::new(i)));
            // weight_priors.push(7.0);
            // objectives.push(Box::new(MatchEEQuatGoals::new(i)));
            // weight_priors.push(5.0);
            objectives.push(Box::new(MinimizeDistanceKeyframeMeanPosition::new(i)));
            weight_priors.push(3.0);
            objectives.push(Box::new(MinimizeDistanceKeyframeMeanOrientation::new(i)));
            weight_priors.push(3.0);
        }
        // objectives.push(Box::new(MinimizeVelocity));
        // weight_priors.push(1.0);
        // objectives.push(Box::new(MinimizeAcceleration));
        // weight_priors.push(1.0);
        // objectives.push(Box::new(MinimizeJerk));
        // weight_priors.push(1.0);
        objectives.push(Box::new(JointLimits));
        weight_priors.push(1.0);
        objectives.push(Box::new(NNSelfCollision));
        weight_priors.push(1.0);
        
        Self {
            objectives,
            num_chains,
            weight_priors,
            lite: false,
            finite_diff_grad: true,
        } // fix this
    }

    pub fn tsr_optimize(num_chains: usize, objective_mode: String) -> Self {
        let mut objectives: Vec<Box<dyn ObjectiveTrait + Send>> = Vec::new();
        let mut weight_priors: Vec<f64> = Vec::new();
        for i in 0..num_chains {
            objectives.push(Box::new(EnvCollision::new(i)));
            if objective_mode == "noECA" {
                weight_priors.push(0.0);
            } else {
                weight_priors.push(1.0);
            }
            objectives.push(Box::new(PlanningTSRError::new(i)));
            weight_priors.push(5.0);
            // objectives.push(Box::new(MatchEEPosGoals::new(i)));
            // weight_priors.push(1.0);
            // objectives.push(Box::new(MatchEEQuatGoals::new(i)));
            // weight_priors.push(1.0);
            
        }
        objectives.push(Box::new(MinimizeVelocity));
        weight_priors.push(7.0);
        objectives.push(Box::new(MinimizeAcceleration));
        weight_priors.push(2.0);
        objectives.push(Box::new(MinimizeJerk));
        weight_priors.push(1.0);
        objectives.push(Box::new(JointLimits));
        weight_priors.push(1.0);
        objectives.push(Box::new(NNSelfCollision));
        weight_priors.push(1.0);
        Self {
            objectives,
            num_chains,
            weight_priors,
            lite: false,
            finite_diff_grad: true,
        }
    }

    pub fn tsr_collision_ik(num_chains: usize, objective_mode: String) -> Self {
        let mut objectives: Vec<Box<dyn ObjectiveTrait + Send>> = Vec::new();
        let mut weight_priors: Vec<f64> = Vec::new();
        for i in 0..num_chains {
            objectives.push(Box::new(MatchEEPosGoals::new(i)));
            weight_priors.push(1.0);
            objectives.push(Box::new(MatchEEQuatGoals::new(i)));
            if objective_mode == "ECA3" {
                weight_priors.push(0.0);
            } else if objective_mode == "ECAA" {
                weight_priors.push(1.0);
            } else {
                weight_priors.push(1.0);
            }
            objectives.push(Box::new(EnvCollision::new(i)));
            if objective_mode == "noECA" {
                weight_priors.push(0.0);
            } else {
                weight_priors.push(1.0);
            }
            objectives.push(Box::new(PlanningTSRError::new(i)));
            weight_priors.push(1.0);
            // objectives.push(Box::new(MatchEEPosGoals::new(i)));
            // weight_priors.push(1.0);
            // objectives.push(Box::new(MatchEEQuatGoals::new(i)));
            // weight_priors.push(1.0);
            
        }
        objectives.push(Box::new(MinimizeVelocity));
        weight_priors.push(7.0);
        objectives.push(Box::new(MinimizeAcceleration));
        weight_priors.push(2.0);
        objectives.push(Box::new(MinimizeJerk));
        weight_priors.push(1.0);
        objectives.push(Box::new(JointLimits));
        weight_priors.push(1.0);
        objectives.push(Box::new(NNSelfCollision));
        weight_priors.push(1.0);
        Self {
            objectives,
            num_chains,
            weight_priors,
            lite: false,
            finite_diff_grad: true,
        }
    }

    pub fn call(&self, x: &[f64], vars: &AgentVars) -> f64 {
        if self.lite {
            self.__call_lite(x, vars)
        } else {
            self.__call(x, vars)
        }
    }

    pub fn gradient(&self, x: &[f64], vars: &AgentVars) -> (f64, Vec<f64>) {
        if self.lite {
            if self.finite_diff_grad {
                self.__gradient_finite_diff_lite(x, vars)
            } else {
                self.__gradient_lite(x, vars)
            }
        } else {
            if self.finite_diff_grad {
                self.__gradient_finite_diff(x, vars)
            } else {
                self.__gradient(x, vars)
            }
        }
    }

    pub fn gradient_finite_diff(&self, x: &[f64], vars: &AgentVars) -> (f64, Vec<f64>) {
        if self.lite {
            self.__gradient_finite_diff_lite(x, vars)
        } else {
            self.__gradient_finite_diff(x, vars)
        }
    }

    fn __call(&self, x: &[f64], vars: &AgentVars) -> f64 {
        let mut out = 0.0;
        let frames = vars.robot.get_frames_immutable(x);
        for i in 0..self.objectives.len() {
            out += self.weight_priors[i] * self.objectives[i].call(x, vars, &frames);
        }
        out
    }

    fn __call_lite(&self, x: &[f64], vars: &AgentVars) -> f64 {
        let mut out = 0.0;
        let poses = vars.robot.get_ee_pos_and_quat_immutable(x);
        for i in 0..self.objectives.len() {
            out += self.weight_priors[i] * self.objectives[i].call_lite(x, vars, &poses);
        }
        out
    }

    fn __gradient(&self, x: &[f64], vars: &AgentVars) -> (f64, Vec<f64>) {
        let mut grad: Vec<f64> = vec![0.; x.len()];
        let mut obj = 0.0;

        let mut finite_diff_list: Vec<usize> = Vec::new();
        let mut f_0s: Vec<f64> = Vec::new();
        let frames_0 = vars.robot.get_frames_immutable(x);
        for i in 0..self.objectives.len() {
            if self.objectives[i].gradient_type() == 1 {
                let (local_obj, local_grad) = self.objectives[i].gradient(x, vars, &frames_0);
                f_0s.push(local_obj);
                obj += self.weight_priors[i] * local_obj;
                for j in 0..local_grad.len() {
                    grad[j] += self.weight_priors[i] * local_grad[j];
                }
            } else if self.objectives[i].gradient_type() == 0 {
                finite_diff_list.push(i);
                let local_obj = self.objectives[i].call(x, vars, &frames_0);
                obj += self.weight_priors[i] * local_obj;
                f_0s.push(local_obj);
            }
        }

        if finite_diff_list.len() > 0 {
            for i in 0..x.len() {
                let mut x_h = x.to_vec();
                x_h[i] += 0.0000001;
                let frames_h = vars.robot.get_frames_immutable(x_h.as_slice());
                for j in &finite_diff_list {
                    let f_h = self.objectives[*j].call(x, vars, &frames_h);
                    grad[i] += self.weight_priors[*j] * ((-f_0s[*j] + f_h) / 0.0000001);
                }
            }
        }

        (obj, grad)
    }

    fn __gradient_lite(&self, x: &[f64], vars: &AgentVars) -> (f64, Vec<f64>) {
        let mut grad: Vec<f64> = vec![0.; x.len()];
        let mut obj = 0.0;

        let mut finite_diff_list: Vec<usize> = Vec::new();
        let mut f_0s: Vec<f64> = Vec::new();
        let poses_0 = vars.robot.get_ee_pos_and_quat_immutable(x);
        for i in 0..self.objectives.len() {
            if self.objectives[i].gradient_type() == 1 {
                let (local_obj, local_grad) = self.objectives[i].gradient_lite(x, vars, &poses_0);
                f_0s.push(local_obj);
                obj += self.weight_priors[i] * local_obj;
                for j in 0..local_grad.len() {
                    grad[j] += self.weight_priors[i] * local_grad[j];
                }
            } else if self.objectives[i].gradient_type() == 0 {
                finite_diff_list.push(i);
                let local_obj = self.objectives[i].call_lite(x, vars, &poses_0);
                obj += self.weight_priors[i] * local_obj;
                f_0s.push(local_obj);
            }
        }

        if finite_diff_list.len() > 0 {
            for i in 0..x.len() {
                let mut x_h = x.to_vec();
                x_h[i] += 0.0000001;
                let poses_h = vars.robot.get_ee_pos_and_quat_immutable(x_h.as_slice());
                for j in &finite_diff_list {
                    let f_h = self.objectives[*j].call_lite(x, vars, &poses_h);
                    grad[i] += self.weight_priors[*j] * ((-f_0s[*j] + f_h) / 0.0000001);
                }
            }
        }

        (obj, grad)
    }

    fn __gradient_finite_diff(&self, x: &[f64], vars: &AgentVars) -> (f64, Vec<f64>) {
        let mut grad: Vec<f64> = vec![0.; x.len()];
        let mut f_0 = self.call(x, vars);

        for i in 0..x.len() {
            let mut x_h = x.to_vec();
            x_h[i] += 0.000001;
            let f_h = self.call(x_h.as_slice(), vars);
            grad[i] = (-f_0 + f_h) / 0.000001;
        }

        (f_0, grad)
    }

    fn __gradient_finite_diff_lite(&self, x: &[f64], vars: &AgentVars) -> (f64, Vec<f64>) {
        let mut grad: Vec<f64> = vec![0.; x.len()];
        let mut f_0 = self.call(x, vars);

        for i in 0..x.len() {
            let mut x_h = x.to_vec();
            x_h[i] += 0.000001;
            let f_h = self.__call_lite(x_h.as_slice(), vars);
            grad[i] = (-f_0 + f_h) / 0.000001;
        }

        (f_0, grad)
    }
}
