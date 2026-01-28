use clap::{Parser, Subcommand};
use fast_insight_engine::{Dataset, InsightEngine};
use std::fs;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "fast-insight-engine")]
#[command(author = "Hummer Team")]
#[command(version = "0.1.0")]
#[command(about = "A fast data processing engine for generating insights", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Load and analyze a CSV file
    Csv {
        /// Name for the dataset
        #[arg(short, long)]
        name: String,

        /// Path to CSV file
        #[arg(short, long)]
        file: PathBuf,

        /// Field to analyze (optional)
        #[arg(short = 'a', long)]
        analyze: Option<String>,
    },

    /// Load and analyze a JSON file
    Json {
        /// Name for the dataset
        #[arg(short, long)]
        name: String,

        /// Path to JSON file
        #[arg(short, long)]
        file: PathBuf,

        /// Field to analyze (optional)
        #[arg(short = 'a', long)]
        analyze: Option<String>,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let mut engine = InsightEngine::new();

    match cli.command {
        Commands::Csv { name, file, analyze } => {
            let content = fs::read_to_string(&file)?;
            let dataset = Dataset::from_csv(name.clone(), &content)?;

            println!("Loaded CSV dataset '{}' with {} records", name, dataset.len());
            println!("Fields: {:?}", dataset.get_field_names());

            engine.add_dataset(dataset);

            if let Some(field) = analyze {
                if let Some(stats) = engine.compute_stats(&name, &field) {
                    print_stats(&stats);
                } else {
                    println!("Could not compute statistics for field '{}'", field);
                }
            }
        }

        Commands::Json { name, file, analyze } => {
            let content = fs::read_to_string(&file)?;
            let dataset = Dataset::from_json(name.clone(), &content)?;

            println!("Loaded JSON dataset '{}' with {} records", name, dataset.len());
            println!("Fields: {:?}", dataset.get_field_names());

            engine.add_dataset(dataset);

            if let Some(field) = analyze {
                if let Some(stats) = engine.compute_stats(&name, &field) {
                    print_stats(&stats);
                } else {
                    println!("Could not compute statistics for field '{}'", field);
                }
            }
        }
    }

    Ok(())
}

fn print_stats(stats: &fast_insight_engine::Statistics) {
    println!("\n=== Statistics for '{}' ===", stats.field);
    println!("Count: {}", stats.count);
    println!("Sum:   {:.2}", stats.sum);
    println!("Mean:  {:.2}", stats.mean);
    println!("Min:   {:.2}", stats.min);
    println!("Max:   {:.2}", stats.max);
}
