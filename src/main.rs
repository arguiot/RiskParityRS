use std::collections::HashMap;
use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader};

use clarabel::algebra::CscMatrix;
use clarabel::solver::{
    DefaultSettings, DefaultSolver, IPSolver, SolverStatus,
    SupportedConeT::{ExponentialConeT, NonnegativeConeT, ZeroConeT},
};
use ndarray::{Array1, Array2};

#[derive(Debug, Clone)]
pub struct ConstraintData {
    pub max_weight: Option<f64>,
    pub sum: Option<f64>,
    pub assets: Vec<String>,
}

/// Main RiskParity struct
pub struct RiskParity {
    pub data: Vec<Vec<f64>>,
    pub asset_names: Vec<String>,
    pub frequency: usize,
    pub returns_data: bool,
    pub log_returns: bool,
    pub cov_matrix: Option<Array2<f64>>,
    pub processed_max_weights: HashMap<String, ConstraintData>,
    pub asset_constraints: HashMap<String, (f64, f64)>,
}

impl RiskParity {
    pub fn new(
        data: Vec<Vec<f64>>,
        mut asset_names: Vec<String>,
        frequency: usize,
        returns_data: bool,
        log_returns: bool,
    ) -> Self {
        // asset_names.sort_unstable();
        Self {
            data,
            asset_names,
            frequency,
            returns_data,
            log_returns,
            cov_matrix: None,
            processed_max_weights: HashMap::new(),
            asset_constraints: HashMap::new(),
        }
    }

    /// Compute sample covariance => shift-diagonal fix if non-PSD
    pub fn compute_sample_cov(&mut self, shift_delta: f64) {
        let returns = if self.returns_data {
            array_from_2d(&self.data)
        } else {
            let rets2d = prices_to_returns(&self.data, self.log_returns);
            array_from_2d(&rets2d)
        };
        let cov = sample_cov(&returns, self.frequency);
        let fixed = fix_nonpositive_semidefinite_diag(&cov, shift_delta);
        self.cov_matrix = Some(fixed);
    }

    /// Minimal constraint processing: fill out (0, max) for each asset
    pub fn process_constraints(&mut self) -> Result<(), String> {
        let default_max = self
            .processed_max_weights
            .get("*")
            .and_then(|cd| cd.max_weight)
            .unwrap_or(1.0);

        for a in &self.asset_names {
            let a_lc = a.to_lowercase();
            if !self.asset_constraints.contains_key(&a_lc) {
                self.asset_constraints.insert(a_lc, (0.0, default_max));
            }
        }
        let sum_of_max: f64 = self.asset_constraints.values().map(|(_mn, mx)| mx).sum();
        if sum_of_max < 1.0 {
            return Err(format!(
                "Sum of max weights < 1.0 => possibly infeasible (sum={:.3}).",
                sum_of_max
            ));
        }
        Ok(())
    }

    /// Solve a single portfolio problem using:
    ///   w in R^N, log_w in R^N, k in R^1, alpha in R^1,
    ///   3n slack variables for ExpCone:
    ///     For i=0..n-1 => (u_i, v_i, wslack_i)
    ///
    /// Constraints:
    ///   1) sum(w) = k
    ///   2) w_i <= max_i * k
    ///   3) sum(log_w) - alpha = 1, alpha >= 0  => sum(log_w) >= 1
    ///   4) ExpCone(u_i, v_i, wslack_i) => v_i>0, wslack_i >= v_i * exp(u_i/v_i),
    ///      plus linear tie: u_i = M * log_w_i, v_i= M, wslack_i = M * w_i
    ///      => w_i = exp(log_w_i) if the solver saturates the inequality.
    ///   5) Objective => w^T cov w
    ///
    /// We'll do dimension = main_vars + slack_exp. main_vars = 2n + 2 => (w, log_w, k, alpha).
    /// Slack_exp = 3n => total = 2n+2 + 3n = 5n+2.
    pub fn optimize(&self) -> Result<Vec<f64>, String> {
        let n = self.asset_names.len();
        let cov = self
            .cov_matrix
            .as_ref()
            .ok_or_else(|| "Covariance not set!".to_string())?;

        //------------------------------------------
        // 1) Build Q for objective => w^T Cov w
        // We'll store w in the first n variables
        // We'll do dimension = 5n+2
        // Indices:
        //   w[i] => i in [0..n-1]
        //   log_w[i] => i in [n..2n-1]
        //   k => index = 2n
        //   alpha => index = 2n+1
        //   Slack for ExpCone => 3n => for each i, (u_i, v_i, wslack_i)
        //------------------------------------------
        let big_dim = 5 * n + 2;

        // We'll place the NxN block for w in the top-left of big_dim x big_dim.
        // That means row in [0..n), col in [0..n).
        let mut row_inds = Vec::new();
        let mut col_inds = Vec::new();
        let mut vals = Vec::new();
        for i in 0..n {
            for j in 0..n {
                let val = 2.0 * cov[[i, j]]; // factor 2 => replicate w^T C w
                if val.abs() > 1e-14 {
                    row_inds.push(i);
                    col_inds.push(j);
                    vals.push(val);
                }
            }
        }
        let Q = CscMatrix::new_from_triplets(big_dim, big_dim, row_inds, col_inds, vals);

        // c => 0
        let c = vec![0.0; big_dim];

        //------------------------------------------
        // 2) Build linear constraints in A x + s = b
        // We'll define them row by row:
        //
        //   Row 0: sum(w_i) - k = 0   => ZeroCone
        //   Row 1: sum(log_w_i) - alpha = 1 => ZeroCone
        //   alpha >= 0 => in NonnegativeCone(1)
        //
        //   Then w_i <= max_i * k => w_i - max_i*k <= 0 => we'll store them in Nonnegative.
        //     We define row i => w_i - max_i*k + slack_i=0 => slack_i>=0
        //
        //   Next, we do ExpCone for each asset => 3 slack variables => tie them with linear eqs:
        //     u_i - M * log_w_i = 0
        //     v_i - M = 0
        //     wslack_i - M * w_i = 0
        //   Then (u_i, v_i, wslack_i) in ExpConeT()
        //------------------------------------------

        // We'll assign indices for these additional variables:
        //   w[i] => i
        //   log_w[i] => n + i
        //   k => 2n
        //   alpha => 2n+1
        //   For each i => u_i => 2n+2 + 3i, v_i => 2n+2 + 3i+1, wslack_i => 2n+2 + 3i+2

        let k_index = 2 * n;
        let alpha_index = 2 * n + 1;
        let slack_exp_start = 2 * n + 2;

        // We'll build the rows in triplet form:
        let mut A_rows = Vec::new();
        let mut A_cols = Vec::new();
        let mut A_vals = Vec::new();
        let mut b = Vec::new();

        // We'll keep a row_count to track which row we're on
        let mut row_count = 0;

        //------------------------------------------
        // (a) Row 0 => sum(w) - k = 0
        //------------------------------------------
        let row_sum_w = row_count;
        row_count += 1;
        for i in 0..n {
            A_rows.push(row_sum_w);
            A_cols.push(i); // w_i
            A_vals.push(1.0);
        }
        // subtract k
        A_rows.push(row_sum_w);
        A_cols.push(k_index);
        A_vals.push(-1.0);

        // b for row_sum_w
        b.push(0.0);

        //------------------------------------------
        // (b) Row 1 => sum(log_w) - alpha = 1 => ZeroCone
        //------------------------------------------
        let row_sum_log = row_count;
        row_count += 1;
        for i in 0..n {
            A_rows.push(row_sum_log);
            A_cols.push(n + i); // log_w_i
            A_vals.push(1.0);
        }
        // -alpha
        A_rows.push(row_sum_log);
        A_cols.push(alpha_index);
        A_vals.push(-1.0);

        b.push(1.0);

        //------------------------------------------
        // (c) alpha >=0 => We'll handle that as part of NonnegativeCone(??)
        // We'll do that by adding no row but 1 dimension in the Nonnegative block.
        //------------------------------------------

        //------------------------------------------
        // (d) w_i <= max_i * k => "w_i - max_i*k <=0" => let’s do w_i - max_i*k + slack_i=0 => slack_i>=0
        //------------------------------------------
        // We'll add n such rows
        let row_wmax_start = row_count;
        for i in 0..n {
            let row_i = row_wmax_start + i;
            let asset_lc = self.asset_names[i].to_lowercase();
            let max_i = self
                .asset_constraints
                .get(&asset_lc)
                .map(|(_min, mx)| mx)
                .unwrap_or(&1.0);

            // w_i
            A_rows.push(row_i);
            A_cols.push(i);
            A_vals.push(1.0);

            // -max_i*k
            A_rows.push(row_i);
            A_cols.push(k_index);
            A_vals.push(-max_i);

            // b[row_i] = 0
            b.push(0.0);
        }
        row_count += n;

        //------------------------------------------
        // (e) Exponential Cones
        // For each i, define:
        //   u_i = 1000 * log_w_i  => row eq
        //   v_i = 1000           => row eq
        //   wslack_i = 1000 * w_i => row eq
        // Then (u_i, v_i, wslack_i) in ExpConeT()
        //------------------------------------------
        let M = 1000.0;

        let row_exp_start = row_count;
        // We need 3 rows per i
        for i in 0..n {
            let row_u = row_exp_start + 3 * i;
            let row_v = row_u + 1;
            let row_wslack = row_u + 2;

            // Indices for these slack variables:
            let u_i = slack_exp_start + 3 * i;
            let v_i = u_i + 1;
            let wslack_i = u_i + 2;

            // (u_i) - M * log_w_i = 0
            A_rows.push(row_u);
            A_cols.push(u_i);
            A_vals.push(1.0);

            A_rows.push(row_u);
            A_cols.push(n + i); // log_w_i at index n+i
            A_vals.push(-M);

            b.push(0.0);

            // (v_i) = M
            A_rows.push(row_v);
            A_cols.push(v_i);
            A_vals.push(1.0);

            // b[row_v] = M
            b.push(M);

            // (wslack_i) - M * w_i = 0
            A_rows.push(row_wslack);
            A_cols.push(wslack_i);
            A_vals.push(1.0);

            A_rows.push(row_wslack);
            A_cols.push(i); // w_i
            A_vals.push(-M);

            b.push(0.0);
        }
        row_count += 3 * n;

        // (f) A row to anchor alpha in the NonnegativeCone
        let row_alpha_nonneg = row_count;
        row_count += 1;

        // alpha_index = 2n+1
        A_rows.push(row_alpha_nonneg);
        A_cols.push(alpha_index);
        A_vals.push(1.0);

        // b for this row => alpha = 0 + slack (which is alpha itself)
        b.push(0.0);

        //------------------------------------------
        // Summarize total # of linear rows
        //------------------------------------------
        let total_rows = row_count; // 2 + n + 3n = 3n + n + 2 = 4n + 2 ... actually it's 1 row sum_w, 1 row sum_log, n rows w_i <= max_i*k => n, plus 3n for exp => 4n+2

        // Build the final A
        let A = CscMatrix::new_from_triplets(total_rows, big_dim, A_rows, A_cols, A_vals);

        //------------------------------------------
        // 3) Build the cones array
        //------------------------------------------
        // We have:
        //  - ZeroConeT(2) for row0..1 => sum(w)=k, sum(log_w)-alpha=1
        //  - NonnegativeConeT(n+1) for alpha>=0 plus the n constraints w_i<=max_i*k
        //  - n ExponentialConeT() for each triple
        //
        // Let's define them in the correct order. The linear rows are stacked as follows:
        //   row0 => sum(w)=k
        //   row1 => sum(log_w)-alpha=1
        //   row2..(2+n-1) => w_i - max_i*k=0 => i in [0..n-1]
        //   rowExp => 3n for the exp constraints
        // dimension of zero cone = 2
        // dimension of nonneg cone = n+1  (1 for alpha + n for w_i - max_i*k slack)
        // dimension for each exp cone = 3
        //
        // So total = 2 + (n+1) + 3n = 4n+3. That must match total_rows. Indeed total_rows= 2 + n + 3n = 4n+2,
        // but we have an extra row for alpha? Actually alpha>=0 doesn't add a row. It's just a variable in the Nonnegative cone.
        // We do have the row for sum(log_w)-alpha=1 => that is in ZeroConeT.
        // So final is:
        //   ZeroConeT(2) for rows [0..2)
        //   NonnegativeConeT(n) for rows [2..2+n) => w_i - max_i*k
        //   BUT alpha >=0 is also in that Nonnegative block => so dimension = n+1
        //   Then n ExponentialConeT => dimension=3 each => total 3n
        //
        // But Clarabel organizes them in order. So we have 2 rows for Zero, n rows for Nonneg => total 2+n = 2 + n. We must also put alpha in that same Nonnegative block. That doesn't add a row, it just means alpha is in that cone. So the dimension is n+1 for the Nonnegative cone.
        // Then n ExpCone => dimension= 3 each => 3n. total = (2 + (n+1) + 3n )= 4n+3 but we only have 4n+2 rows.
        // We see a mismatch of 1 because alpha >=0 doesn't add a row. So the total row_count= 4n+2 is correct for the linear constraints. The Nonnegative cone has dimension = n+1, but that means an extra "slot" in the cone for alpha.
        // We'll do this by telling Clarabel that the second cone has dimension n+1, which means the solver expects that after we've used up 2 rows in ZeroCone, the next n+1 "rows" are for the Nonnegative cone. But we only physically have n rows in the matrix for w_i - max_i*k.
        // The alpha variable is "a free variable" that we want in Nonnegative. Usually that means a separate row 0= alpha? Actually let's do "A x + s = b" => we want alpha in the Nonneg? It's simpler to define "alpha" as part of the same Nonnegative block but no row. So let's show how to do it in clarabel.
        //
        // We'll do:
        //   cones[0] => ZeroConeT(2)
        //   cones[1] => NonnegativeConeT(n+1)
        //   cones[2..(2+n)] => n ExpConeT
        //
        // That means total "conic dimension" = 2 + (n+1) + 3n = 4n+3. But we only have row_count=4n+2. Usually, the extra dimension for alpha is a "free" row that doesn't appear in A. So effectively there's 1 "row" in the Nonnegative block that doesn't tie to the matrix. This is how Clarabel organizes it.
        //
        // We'll do an approach:
        //   - Zero(2) covers row0..1
        //   - Nonnegative(n) covers row2..2+n => these are w_i - max_i*k constraints
        //   - Then we *also* add alpha to Nonnegative => dimension n+1 total => so that's 1 more dimension.
        //   - Then we add n ExpCone => each dimension 3 => total 3n.
        //
        // This should work so long as clarabel is fine with an "extra variable" in that cone not associated to any row. Typically, it is. We'll test.
        //------------------------------------------

        let mut cones = Vec::new();
        cones.push(ZeroConeT(2)); // row0..1 => sum(w)=k, sum(log_w)-alpha=1
        cones.push(NonnegativeConeT(n + 1)); // n rows for w_i - max_i*k, plus alpha variable

        // Then add n ExpCones
        for _ in 0..n {
            cones.push(ExponentialConeT());
        }

        //------------------------------------------
        // 4) Build and solve
        //------------------------------------------
        let settings = DefaultSettings {
            verbose: true,
            ..Default::default()
        };
        let mut solver = DefaultSolver::new(&Q, &c, &A, &b, &cones, settings);

        solver.solve();
        if solver.info.status != SolverStatus::Solved {
            return Err(format!("Clarabel solver failed: {:?}", solver.info.status));
        }

        // x in R^(5n+2), so let's parse out w, log_w, k, alpha
        let x_sol = solver.solution.x.clone();
        let mut wvals = Vec::with_capacity(n);
        for i in 0..n {
            wvals.push(x_sol[i]);
        }
        // Optionally parse log_w, k, alpha if you want them
        let _log_wvals: Vec<f64> = (0..n).map(|i| x_sol[n + i]).collect();
        let _k = x_sol[k_index];
        let _alpha = x_sol[alpha_index];

        // Possibly normalize w if Python does after solve.
        // The python code typically does => final w = w / sum(w). But we have sum(w)=k, and if k isn't 1, we might.
        // If you want final sum=1, do:
        let sum_w: f64 = wvals.iter().sum();
        if sum_w.abs() > 1e-14 {
            for i in 0..n {
                wvals[i] /= sum_w;
            }
        }

        Ok(wvals)
    }

    /// High-level user function
    pub fn get_weights(&mut self, shift_delta: f64) -> Result<Vec<f64>, String> {
        if self.cov_matrix.is_none() {
            self.compute_sample_cov(shift_delta);
        }
        self.process_constraints()?;
        self.optimize()
    }
}

// ----------------------------
// Utility Functions
// ----------------------------

fn array_from_2d(data: &Vec<Vec<f64>>) -> Array2<f64> {
    let rows = data.len();
    if rows == 0 {
        return Array2::zeros((0, 0));
    }
    let cols = data[0].len();
    let mut arr = Array2::<f64>::zeros((rows, cols));
    for i in 0..rows {
        for j in 0..cols {
            arr[[i, j]] = data[i][j];
        }
    }
    arr
}

fn prices_to_returns(data: &Vec<Vec<f64>>, log_returns: bool) -> Vec<Vec<f64>> {
    if data.len() < 2 {
        return vec![];
    }
    let mut out = Vec::with_capacity(data.len() - 1);
    for i in 1..data.len() {
        let mut row = Vec::with_capacity(data[i].len());
        for j in 0..data[i].len() {
            let p0 = data[i - 1][j];
            let p1 = data[i][j];
            if log_returns {
                row.push((p1 / p0).ln());
            } else {
                row.push(p1 / p0 - 1.0);
            }
        }
        out.push(row);
    }
    out
}

/// sample_cov => frequency * (1/(T-1)) * centered^T * centered
fn sample_cov(returns: &Array2<f64>, frequency: usize) -> Array2<f64> {
    let (t, n) = returns.dim();
    if t < 2 {
        return Array2::zeros((n, n));
    }
    let mean_by_col = returns.mean_axis(ndarray::Axis(0)).unwrap();
    let mut centered = returns.clone();
    for i in 0..t {
        for j in 0..n {
            centered[[i, j]] -= mean_by_col[j];
        }
    }
    let denom = (t - 1) as f64;
    let mat = centered.t().dot(&centered) / denom;
    mat * (frequency as f64)
}

/// *Diagonal Shift* fix => no eigen-decomposition
fn fix_nonpositive_semidefinite_diag(matrix: &Array2<f64>, shift_delta: f64) -> Array2<f64> {
    let mut fixed = matrix.clone();
    let n = matrix.shape()[0];
    for i in 0..n {
        fixed[[i, i]] += shift_delta;
    }
    fixed
}

// ----------------------------
// Main (demonstration)
// ----------------------------
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    let prices_data = if args.len() > 1 {
        let path = &args[1];
        println!("Reading CSV from {path}");
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut rows = Vec::new();
        for line in reader.lines() {
            let l = line?;
            let cols: Vec<&str> = l.split(',').collect();
            let mut rowvals = Vec::with_capacity(cols.len());
            for c in cols {
                rowvals.push(c.trim().parse::<f64>()?);
            }
            rows.push(rowvals);
        }
        rows
    } else {
        println!("No CSV provided => using fallback data...");
        vec![
            vec![100.0, 50.0, 30.0],
            vec![101.0, 51.0, 29.5],
            vec![102.0, 49.0, 31.0],
            vec![100.0, 52.0, 32.0],
            vec![99.0, 50.5, 33.0],
        ]
    };

    let asset_names = vec!["btc".to_string(), "eth".to_string(), "bnb".to_string()];
    let mut rp = RiskParity::new(
        prices_data,
        asset_names,
        365,   // frequency
        false, // returns_data
        false, // log_returns
    );

    // Suppose we want a global max=0.8
    rp.processed_max_weights.insert(
        "*".to_string(),
        ConstraintData {
            max_weight: Some(0.8),
            sum: None,
            assets: vec![],
        },
    );

    // Now solve with e.g. 1e-5 shift
    match rp.get_weights(1e-5) {
        Ok(w) => {
            println!("Final weights:");
            for (i, weight) in w.iter().enumerate() {
                println!("  {}: {:.4}", rp.asset_names[i], weight);
            }
        }
        Err(e) => println!("Solver error: {}", e),
    }
    Ok(())
}
