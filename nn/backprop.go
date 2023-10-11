package nn

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

func (nn *NN) Backprop() {
	var (
		activation = *nn.ActivationFunc
		// lossHist   []float64
	)

	for i := 0; i < nn.Epochs; i++ {
		// compute output with current w + b
		// then compute loss & backprop
		hiddenOutput, err := computeOutput(
			nn.XTrain,
			nn.Wh,
			nn.Bh,
			activation,
		)
		if err != nil {
			fmt.Printf("error computing hidden output: %v", err)
		}

		yHat, err := computeOutput(
			hiddenOutput,
			nn.Wo,
			nn.Bo,
			activation,
		)
		if err != nil {
			fmt.Printf("error computing yHat: %v", err)
		}

		mse := meanSquaredError(nn.YTrain, yHat)
		fmt.Println(mse)

	}

}

func computeOutput(arr, w, b *mat.Dense, activationFunc func(float64) float64) (*mat.Dense, error) {
	// Check if any of the input matrices is nil
	if arr == nil || w == nil || b == nil {
		return nil, fmt.Errorf("Input matrices cannot be nil")
	}

	// Check input dimensions
	arrRows, arrCols := arr.Dims()
	wRows, wCols := w.Dims()
	bRows, bCols := b.Dims()

	if arrCols != wRows || bCols != wCols {
		return nil, fmt.Errorf("Matrix dimension mismatch: arr[%d, %d], w[%d, %d], b[%d, %d]", arrRows, arrCols, wRows, wCols, bRows, bCols)
	}

	// Compute the dot product between the input matrix 'arr' and the weight matrix 'w'
	var product mat.Dense
	product.Mul(arr, w)

	// Check dimensions of product and bias
	productRows, productCols := product.Dims()
	if productCols != bCols {
		return nil, fmt.Errorf("Matrix dimension mismatch: product[%d, %d], b[%d, %d]", productRows, productCols, bRows, bCols)
	}

	// Add the bias matrix 'b' to the product
	var result mat.Dense
	result.Add(&product, b)

	// Apply the activation function to the result
	applyActivation(&result, activationFunc)

	return &result, nil
}

func applyActivation(m *mat.Dense, f func(float64) float64) {
	r, c := m.Dims()
	data := m.RawMatrix().Data
	for i := 0; i < r*c; i++ {
		data[i] = f(data[i])
	}
}

func meanSquaredError(y, yHat *mat.Dense) float64 {
	var sum float64
	r, c := y.Dims()

	for row := 0; row < r; row++ {
		for col := 0; col < c; col++ {
			diff := y.At(row, col) - yHat.At(row, col)
			sum += (diff * diff)
		}
	}
	return sum / float64((r * c))
}
