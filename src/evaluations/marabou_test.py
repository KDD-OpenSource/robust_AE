import maraboupy
import os
import torch
#import Marabou
from maraboupy import Marabou
from maraboupy import MarabouCore

from .evaluation import evaluation


class marabou_test:
    def __init__(
        self, eval_inst: evaluation, name: str = "marabou_test"
    ):
        self.name = name
        self.evaluation = eval_inst

    def evaluate(self, dataset, algorithm):
        # model one network but with 2 inputs (like a small batch).
        # -> simulates two networks
        randomInput = torch.randn(2,algorithm.topology[0])
        onnx_folder = os.path.join('./models/onnx_models/',
            str(algorithm.name)+'_' + dataset.name)
        os.makedirs(onnx_folder, exist_ok = True)
        torch.onnx.export(algorithm.module.get_neural_net()[:3], randomInput.float(),
                os.path.join(onnx_folder, 'saved_algorithm.onnx'))
        network = Marabou.read_onnx(
                os.path.join(onnx_folder, 'saved_algorithm.onnx'),
                outputName = '7')
        network.saveQuery('./models/marabou_models/saved_algorithm_marabou')
        loaded_network = maraboupy.Marabou.load_query(
                './models/marabou_models/saved_algorithm_marabou')

        # add equations for input and output
        numOutputVars = loaded_network.getNumOutputVariables()
        for ind1, ind2 in zip(
                range(int(numOutputVars/2)),
                range(int(numOutputVars/2), numOutputVars)
                ):
            outputInd1 = loaded_network.outputVariableByIndex(ind1)
            outputInd2 = loaded_network.outputVariableByIndex(ind2)
            equation = MarabouCore.Equation(MarabouCore.Equation.EQ)
            # test for equality
            equation.addAddend(1, outputInd1)
            equation.addAddend(-1, outputInd2)
            loaded_network.addEquation(equation)

        # add lower and upper bounds
        numInputVars = loaded_network.getNumInputVariables()
        for ind1 in range(int(numInputVars/2)):
            input_ind = loaded_network.inputVariableByIndex(ind1)
            loaded_network.setLowerBound(input_ind, -1)
            loaded_network.setUpperBound(input_ind, -0.8)

        for ind2 in range(int(numInputVars/2), numInputVars):
            input_ind = loaded_network.inputVariableByIndex(ind2)
            loaded_network.setLowerBound(input_ind, 0.7)
            loaded_network.setUpperBound(input_ind, 1)

        options = Marabou.createOptions(verbosity = 2)
        solutions, stats = maraboupy.MarabouCore.solve(loaded_network, options)


        #input1, output1, input2, output2 = extract_solution_points(solutions,
                #loaded_network)
        #import pdb; pdb.set_trace()

def extract_solution_points(solutions, loaded_network):
    numInputVars = loaded_network.getNumInputVariables()
    inputpoint1 = []
    for ind1 in range(int(numInputVars/2)):
        input_ind = loaded_network.inputVariableByIndex(ind1)
        inputpoint1.append(solutions[input_ind])

    inputpoint2 = []
    for ind2 in range(int(numInputVars/2), numInputVars):
        input_ind = loaded_network.inputVariableByIndex(ind2)
        inputpoint2.append(solutions[input_ind])

    numOutputVars = loaded_network.getNumOutputVariables()
    outputpoint1 = []
    for ind1 in range(int(numOutputVars/2)):
        output_ind = loaded_network.outputVariableByIndex(ind1)
        outputpoint1.append(solutions[output_ind])

    outputpoint2 = []
    for ind2 in range(int(numOutputVars/2), numOutputVars):
        output_ind = loaded_network.outputVariableByIndex(ind2)
        outputpoint2.append(solutions[output_ind])

    return inputpoint1, outputpoint1, inputpoint2, outputpoint2
