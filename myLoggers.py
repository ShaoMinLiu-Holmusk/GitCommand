import logging

def getLogger(loggerName:str, fileName:str,
              level = logging.DEBUG,
              format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'):

    loggerName = logging.getLogger(loggerName)
    loggerName.setLevel(level)
    myFormat = logging.Formatter(format)
    myHandler = logging.FileHandler(fileName, mode='a')
    myHandler.setFormatter(myFormat)
    loggerName.addHandler(myHandler)

    return loggerName

HyperTuner_objGenerator = getLogger('HyperTuner_objGenerator', 'logs/HyperTuner_objGenerator.log')
HyperTuner_objective = getLogger('HyperTuner_objective', 'logs/HyperTuner_objective.log')
myNeural = getLogger('myNeural', 'logs/myNeural.log')
myNeuralHyperTunner = getLogger('myNeuralHyperTunner', 'logs/myNeuralHyperTunner.log')