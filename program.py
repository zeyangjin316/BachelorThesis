import logging
from copula_method.two_step_model import TwoStepModel
from reader import Reader

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.FileHandler('model.log', mode='w'),  # Add file handler
        logging.StreamHandler()  # Keep console output as well
    ]
)
logging.getLogger('rpy2').setLevel(logging.INFO)


def main():
    # Initialize and run the model
    test = TwoStepModel(n_samples=1000)
    test.fit()
    samples = test.sample()
    print(samples.head(10))
    print(samples.tail(10))
    test.evaluate_energy_score(samples)

if __name__ == "__main__":
    main()