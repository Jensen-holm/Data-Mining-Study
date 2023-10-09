package nn

var ActivationMap = map[string]func(){
	"sigmoid": Sigmoid,
	"tanh":    Tanh,
	"relu":    Relu,
}

func Sigmoid() {}

func SigmoidPrime() {}

func Tanh() {}

func TanhPrime() {}

func Relu() {}

func ReluPrime() {}
