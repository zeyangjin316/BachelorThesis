import logging
from copula_method.two_step_model import TwoStepModel
from data_handling import Reader

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
    test = TwoStepModel(split_point=0.999)
    test.fit()
    samples = test.sample()
    #print(samples.head(10))
    #print(samples.tail(10))
    test.evaluate(samples)

if __name__ == "__main__":
    main()