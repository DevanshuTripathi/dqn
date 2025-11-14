// train.go
package dqn

import (
	"math/rand"
)

// DQN represents the Deep Q-Learning algorithm.
type DQN struct {
	qNetwork      *QNetwork
	targetNetwork *QNetwork
	replayBuffer  *ReplayBuffer
	gamma         float64
	epsilon       float64
	learningRate  float64
	learnStep     int
	batchSize     int
}

// NewDQN initializes a new DQN instance.
func NewDQN(inputSize, hiddenSize, outputSize, bufferSize int, gamma, epsilon, learningRate float64, activation Activation) *DQN {
	agent := &DQN{
		qNetwork:      NewQNetwork(inputSize, hiddenSize, outputSize, activation),
		targetNetwork: NewQNetwork(inputSize, hiddenSize, outputSize, activation),
		replayBuffer:  NewReplayBuffer(bufferSize),
		gamma:         gamma,
		epsilon:       epsilon,
		learningRate:  learningRate,
		batchSize:     64,
	}

	agent.CopyToTarget()
	return agent
}

// Train trains the Q-network.
func (d *DQN) Train(state, nextState []float64, action int, reward float64, done bool) {

	d.replayBuffer.Add(Experience{
		State:     append([]float64{}, state...),
		NextState: append([]float64{}, nextState...),
		Action:    action,
		Reward:    reward,
		Done:      done,
	})

	if d.replayBuffer.Size() < 2000 {
		return
	}

	batch := d.replayBuffer.Sample(d.batchSize)

	for _, exp := range batch {
		nextQ := d.targetNetwork.Predict(exp.NextState)
		maxNext := Max(nextQ)

		curQ := d.qNetwork.Predict(exp.State)

		// Make a copy for target
		target := append([]float64{}, curQ...)

		if exp.Done {
			target[exp.Action] = exp.Reward
		} else {
			target[exp.Action] = exp.Reward + d.gamma*maxNext
		}

		d.qNetwork.Backward(exp.State, curQ, target, d.learningRate)
	}

	d.learnStep++

	if d.learnStep%2000 == 0 {
		d.CopyToTarget()
	}

	// nextQValues := d.qNetwork.Predict(nextState)
	// maxNextQValue := Max(nextQValues)
	// target := make([]float64, len(nextQValues))
	// copy(target, nextQValues)
	// target[action] = reward
	// if !done {
	// 	target[action] += d.gamma * maxNextQValue
	// }

	// currentQValues := d.qNetwork.Predict(state)
	// // loss := d.qNetwork.Loss(currentQValues, target)

	// d.qNetwork.Backward(state, currentQValues, target, d.learningRate)
}

// EpsilonGreedyPolicy selects an action using epsilon-greedy strategy.
func (d *DQN) EpsilonGreedyPolicy(state []float64, numActions int) int {
	if rand.Float64() < d.epsilon {
		return rand.Intn(numActions)
	}
	qValues := d.qNetwork.Predict(state)
	return Argmax(qValues)
}

// Helper functions

func (d *DQN) CopyToTarget() {
	weights := d.qNetwork.GetWeights()
	d.targetNetwork.SetWeights(weights)
}

// Max returns the maximum value in a slice of float64
func Max(arr []float64) float64 {
	maxVal := arr[0]
	for _, val := range arr {
		if val > maxVal {
			maxVal = val
		}
	}
	return maxVal
}

// Argmax returns the index of the maximum value in a slice of float64
func Argmax(arr []float64) int {
	maxIdx := 0
	maxVal := arr[0]
	for i, val := range arr {
		if val > maxVal {
			maxIdx = i
			maxVal = val
		}
	}
	return maxIdx
}
