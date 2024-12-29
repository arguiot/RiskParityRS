# Risk Parity in Rust

This is a simple implementation of the Risk Parity algorithm in Rust.

Example usage:

```rust
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
```
