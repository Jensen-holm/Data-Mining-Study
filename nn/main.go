package nn

import (
	"fmt"

	"github.com/Jensen-holm/ml-from-scratch/alg"
	"github.com/Jensen-holm/ml-from-scratch/request"
)

type NN struct {
	alg.Alg
	args *NNArgs
}

func New(rp *request.Payload) (alg.Alg, error) {
	// parse the args and make a new NN struct
	nnArgs := make(map[string]interface{})
	params := []string{
		"epochs",
		"activation",
		"hidden_size",
		"learning_rate",
	}

	for _, param := range params {
		i, isIn := rp.Args[param]
		if !isIn {
			return nil, fmt.Errorf("user must specify %s", param)
		}

		switch param {
		case "epochs":
			if val, ok := i.(int); ok {
				nnArgs[param] = val
			} else {
				return nil, fmt.Errorf("expected %s to be an int", param)
			}
		case "activation":
			if val, ok := i.(string); ok {
				nnArgs[param] = ActivationMap[val]
			} else {
				return nil, fmt.Errorf("expected %s to be a string", param)
			}
		case "hidden_size":
			if val, ok := i.(int); ok {
				nnArgs[param] = val
			} else {
				return nil, fmt.Errorf("expected %s to be an int", param)
			}
		case "learning_rate":
			if val, ok := i.(float64); ok {
				nnArgs[param] = val
			} else {
				return nil, fmt.Errorf("expected %s to be a float64", param)
			}
		default:
			return nil, fmt.Errorf("unsupported parameter: %s", param)
		}
	}

	args := NewArgs(nnArgs)
	return &NN{
		args: args,
	}, nil

}
