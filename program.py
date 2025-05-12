import logging
from univariate_models import UnivariateModel
from copula_fitting import CopulaEstimator
from two_step_model import TwoStepModel

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.FileHandler('model.log'),  # Add file handler
        logging.StreamHandler()  # Keep console output as well
    ]
)
logging.getLogger('rpy2').setLevel(logging.INFO)


def main():
    # Initialize and run the model
    test = TwoStepModel()
    test.train()

if __name__ == "__main__":
    main()