import logging
from univariate_models import UnivariateModel

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # Initialize and run the model
    model = UnivariateModel(data_input="data_for_kit.csv", split_point=0.8, method="ARMAGARCH")
    summary = model.run()

    # Access different parts of the summary
    print(f"Model type: {summary['model_type']}")
    print(f"Number of symbols: {summary['data_summary']['n_symbols']}")
    print(f"Best performing symbol: {summary['aggregate_metrics']['best_performing_symbol']}")

    # Get results for a specific symbol
    symbol = "AAPL"
    symbol_results = summary['model_results'][symbol]
    print(f"R² for {symbol}: {symbol_results['evaluation_metrics']['r2']:.3f}")

if __name__ == "__main__":
    name()
