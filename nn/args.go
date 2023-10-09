package nn

type NNArgs struct {
	epochs         int
	hiddenSize     int
	learningRate   float64
	activationFunc func()
}

func NewArgs(argsMap map[string]interface{}) *NNArgs {
	return &NNArgs{
		epochs:         argsMap["epochs"].(int),
		hiddenSize:     argsMap["hidden_size"].(int),
		learningRate:   argsMap["learning_rate"].(float64),
		activationFunc: argsMap["activation"].(func()),
	}
}
