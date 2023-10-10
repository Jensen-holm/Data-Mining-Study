package nn

import "math"

var ActivationMap = map[string]func(float64) float64{
	"sigmoid": Sigmoid,
	"tanh":    Tanh,
	"relu":    Relu,
}

func Sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func SigmoidPrime(x float64) float64 {
	s := Sigmoid(x)
	return s / (1.0 - s)
}

func Tanh(x float64) float64 {
	return math.Tanh(x)
}

func TanhPrime(x float64) float64 {
	return math.Pow((1.0 / math.Cosh(x)), 2)
}

func Relu(x float64) float64 {
	if x > 0 {
		return x
	}
	return 0
}

func ReluPrime(x float64) float64 {
	// maybe want to look into edge case if x == 0
	if x > 0 {
		return 1
	}
	return 0
}
