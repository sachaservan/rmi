// regression.go: Basic linear regression implemented
// with big.Rat for increased accuracy

package rmi

import (
	"math/big"
)

// linear_regression on an array given a certain range from
// start to end (inclusive, inclusive)
// function to compute mean, input: float64 array
func mean(values []*big.Int) *big.Float {

	mean := big.NewFloat(0.0)
	for i := 0; i < len(values); i++ {
		mean.Add(mean, new(big.Float).SetInt(values[i]))
	}

	mean.Quo(mean, big.NewFloat(float64(len(values))))
	return mean
}

// function to compute covariance of two arrays,
// input: float64 arrayX and arrayY, meanX and meanY
func covariance(
	x []*big.Int,
	y []*big.Int,
	meanX *big.Float,
	meanY *big.Float) *big.Float {

	covar := big.NewFloat(0.0)
	for i := 0; i < len(x); i++ {
		termX := new(big.Float).SetInt(x[i])
		termX.Sub(termX, meanX)

		termY := new(big.Float).SetInt(y[i])
		termY.Sub(termY, meanY)

		termXY := new(big.Float).Mul(termX, termY)
		covar.Add(covar, termXY)
	}

	return covar
}

// function to compute variance of array, inp: float64 array1 mean1
func variance(values []*big.Int, meanValue *big.Float) *big.Float {

	variance := big.NewFloat(0.0)
	for i := 0; i < len(values); i++ {
		abs := new(big.Float).SetInt(values[i])
		abs.Sub(abs, meanValue)
		abs.Mul(abs, abs)
		variance.Add(variance, abs)
	}

	return variance
}

// function to compute linar regression coefficients + x intercept
func coefficients(predVars []*big.Int, target []*big.Int) (*big.Float, *big.Float, *big.Float) {

	meanX := mean(predVars)
	meanY := mean(target)

	b1 := covariance(predVars, target, meanX, meanY)
	b1.Quo(b1, variance(predVars, meanX))

	b0 := new(big.Float).Sub(meanY, meanX.Mul(meanX, b1))

	w := new(big.Float).Neg(b0)
	w.Quo(w, b1)

	return b0, b1, w
}
