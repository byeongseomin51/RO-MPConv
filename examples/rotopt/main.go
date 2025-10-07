package main

import (
	"fmt"
	"math"
	"os"
	"rotopt/modules"
	"sort"
	"time"
	"unsafe"

	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/ring"
	"github.com/tuneinsight/lattigo/v5/schemes/ckks"
)

func main() {

	args := os.Args[1:]
	//example :
	//go run . conv parBSGS rotkey

	//== supported args ======================================================================================================================================//
	// basic        : Execution time of RNS-CKKS basic operations (rotation, multiplication, addition). [Fig. 1]
	// conv         : Latency comparison for ResNet convolutions (CONV1, CONV2, CONV3s2, etc.). [Fig. 13]
	// otherConv    : Latency comparison for modern architecture convolutions (CvT, MUSE). [Fig. 13]
	// applications : Convolution latency improvements for MPCNN, AutoFHE, CryptoFace, etc. [Table 4]
	// blueprint    : Extracts the blueprints for each RO-MPConv configuration. [Appendix A]
	// fc           : Performance of parallel BSGS mat-vec mul on a fully connected layer scenario. [Fig. 15]
	// matVecMul    : General performance comparison of BSGS matrix-vector multiplication methods. [Fig. 15]
	// ALL          : If you write ALL or don't write any args, all of the test functions will be started.
	//=========================================================================================================================================================//

	if len(args) == 0 {
		fmt.Println("args set as ALL")
		args = []string{"ALL"}
	} else if len(args) == 1 {
		fmt.Println("args : ", args)
	}

	//CKKS settings
	context := setCKKSEnv() //default CKKS environment
	// context := setCKKSEnvUseParamSet("PN15QP880CI") //Lightest parameter set
	// context := setCKKSEnvUseParamSet("PN16QP1761") // Heavy param set

	//basicOperationTimeTest
	if Contains(args, "basic") || args[0] == "ALL" {
		fmt.Println("\nBasic operation time test started!")
		basicOperationTimeTest(context)
	}

	///////////////////////////////////////
	/////////////RO-MPConv Test////////////
	///////////////////////////////////////

	// Convolution Tests
	if Contains(args, "conv") || args[0] == "ALL" {
		rotOptConvTimeTest(context, 2)
		rotOptConvTimeTest(context, 3)
		rotOptConvTimeTest(context, 4)
		rotOptConvTimeTest(context, 5)
		mulParConvTimeTest(context)
	}

	// Generalization of different AI models
	if Contains(args, "otherConv") || args[0] == "ALL" {
		// Each convolution refers to...
		// CvTConv1, CvTConv2 : convolutional embedding in CvT (Convolutional Vision Transformer) model.
		// MuseConv 			  		: create a multi-scale feature pyramid from a single-scale feature map in MUSE (a model based on Mamba). https://ojs.aaai.org/index.php/AAAI/article/view/32778
		otherMulParConvTimeTest(context)
		otherRotOptConvTimeTest(context, 2)
		otherRotOptConvTimeTest(context, 3)
		otherRotOptConvTimeTest(context, 4)
		otherRotOptConvTimeTest(context, 5)
		otherRotOptConvTimeTest(context, 6)
		otherRotOptConvTimeTest(context, 7)
		otherRotOptConvTimeTest(context, 8)
		otherRotOptConvTimeTest(context, 9)
		otherRotOptConvTimeTest(context, 10)
	}

	// Generalization of different AI models
	if Contains(args, "applications") || args[0] == "ALL" {
		conv_time_dict := conv_time_loader() // result form: map[method_name]map[depth]map[conv_type]map[startLevel]time
		MPCNNTimeTest(context, conv_time_dict)
		autoFHEConvTimeTest(context, conv_time_dict)
		cryptoFaceConvTimeTest(context, conv_time_dict)
		aespaConvTimeTest(context, conv_time_dict)
	}

	// Print Blue Print. Corresponds to Appendix A.
	if Contains(args, "blueprint") || args[0] == "ALL" {
		getBluePrint()
	}

	/////////////////////////////////////
	//Matrix-Vector Multiplication Test//
	/////////////////////////////////////
	//Apply to fully connected layer
	if Contains(args, "fc") || args[0] == "ALL" {
		parBSGSfullyConnectedAccuracyTest(context) //using parallel BSGS matrix-vector multiplication to fully connected layer.
		mulParfullyConnectedAccuracyTest(context)  //conventional
	}
	if Contains(args, "matVecMul") || args[0] == "ALL" {
		for N := 32; N <= 512; N *= 2 {
			parBsgsMatVecMultAccuracyTest(N, context) //proposed
			bsgsMatVecMultAccuracyTest(N, context)    //conventional
		}
	}
}

type customContext struct {
	Params      ckks.Parameters
	Encoder     *ckks.Encoder
	Kgen        *rlwe.KeyGenerator
	Sk          *rlwe.SecretKey
	Pk          *rlwe.PublicKey
	EncryptorPk *rlwe.Encryptor
	EncryptorSk *rlwe.Encryptor
	Decryptor   *rlwe.Decryptor
	Evaluator   *ckks.Evaluator
}

func otherRotOptConvTimeTest(cc *customContext, depth int) {
	fmt.Printf("\nRotation Optimized Convolution (for %d-depth consumed, complex AI model) time test started!\n", depth)

	var convIDs []string

	switch depth {
	case 2:
		convIDs = []string{"CvTConv1", "CvTConv2", "MuseConv"}
	case 3:
		convIDs = []string{"CvTConv1", "CvTConv2", "MuseConv"}
	case 4:
		convIDs = []string{"CvTConv1", "CvTConv2", "MuseConv"}
	case 5:
		convIDs = []string{"CvTConv1", "CvTConv2", "MuseConv"}
	case 6:
		convIDs = []string{"CvTConv2", "MuseConv"}
	case 7:
		convIDs = []string{"MuseConv"}
	case 8:
		convIDs = []string{"MuseConv"}
	case 9:
		convIDs = []string{"MuseConv"}
	case 10:
		convIDs = []string{"MuseConv"}
	default:
		fmt.Printf("Unsupported depth: %d\n", depth)
		return
	}

	iter := 1
	minStartCipherLevel := depth
	maxStartCipherLevel := cc.Params.MaxLevel() //원
	// maxStartCipherLevel := depth

	for index := 0; index < len(convIDs); index++ {

		convID := convIDs[index]
		if convID == "MuseConv" {
			cc = setCKKSEnvUseParamSet("PN15QP880CI")

			maxStartCipherLevel = cc.Params.MaxLevel()
		}

		//register index of rotation
		rots := modules.RotOptConvRegister(convID, depth)
		// fmt.Println(len(rots[0]), len(rots[1]), len(rots[2]), len(rots[0])+len(rots[1])+len(rots[2]))
		// continue

		//rotation key register
		newEvaluator := rotIndexToGaloisEl(int2dTo1d(rots), cc.Params, cc.Kgen, cc.Sk)

		//make rotOptConv instance
		conv := modules.NewrotOptConv(newEvaluator, cc.Encoder, cc.Params, convID, depth)

		// Make input and kernel
		cf := conv.ConvFeature
		plainInput := makeRandomInput(cf.InputDataChannel, cf.InputDataHeight, cf.InputDataWidth)
		plainKernel := makeRandomKernel(cf.KernelNumber, cf.InputDataChannel, cf.KernelSize, cf.KernelSize)

		//Plaintext Convolution
		plainOutput := PlainConvolution2D(plainInput, plainKernel, cf.Stride, 1)

		// Encrypt Input, Encode Kernel
		mulParPackedInput := MulParPacking(plainInput, cf, cc)
		conv.PreCompKernels = EncodeKernel(plainKernel, cf, cc)

		fmt.Printf("=== convID : %s, Depth : %v, CipherLevel : %v ~ %v, iter : %v === \n", convID, depth, Max(minStartCipherLevel, depth), maxStartCipherLevel, iter)
		fmt.Printf("startLevel executionTime(sec)\n")
		// MSE, RE, inf Norm
		var MSEList, REList, infNormList []float64
		for startCipherLevel := Max(minStartCipherLevel, depth); startCipherLevel <= maxStartCipherLevel; startCipherLevel++ {
			plain := ckks.NewPlaintext(cc.Params, startCipherLevel)
			cc.Encoder.Encode(mulParPackedInput, plain)
			inputCt, _ := cc.EncryptorSk.EncryptNew(plain)

			var totalForwardTime time.Duration
			//Conv Foward
			for i := 0; i < iter; i++ {
				//Convolution start
				start := time.Now()
				encryptedOutput := conv.Foward(inputCt)
				end := time.Now()
				totalForwardTime += end.Sub(start)

				// MSE, RE, inf Norm
				FHEOutput := UnMulParPacking(encryptedOutput, cf, cc)
				scores := MSE_RE_infNorm(plainOutput, FHEOutput)
				MSEList = append(MSEList, scores[0])
				REList = append(REList, scores[1])
				infNormList = append(infNormList, scores[2])
			}

			//Print Elapsed Time
			avgForwardTime := float64(totalForwardTime.Nanoseconds()) / float64(iter) / 1e9
			fmt.Printf("%v %v \n", startCipherLevel, avgForwardTime)
		}
		// Average Acc, Recall, F1 score
		MSEMin, MSEMax, MSEAvg := minMaxAvg(MSEList)
		REMin, REMax, REAvg := minMaxAvg(REList)
		infNormMin, infNormMax, infNormAvg := minMaxAvg(infNormList)

		fmt.Printf("MSE (Mean Squared Error)   : Min = %.2e, Max = %.2e, Avg = %.2e\n", MSEMin, MSEMax, MSEAvg)
		fmt.Printf("Relative Error             : Min = %.2e, Max = %.2e, Avg = %.2e\n", REMin, REMax, REAvg)
		fmt.Printf("Infinity Norm (L-infinity) : Min = %.2e, Max = %.2e, Avg = %.2e\n", infNormMin, infNormMax, infNormAvg)
		fmt.Println()
	}
}
func otherMulParConvTimeTest(cc *customContext) {
	fmt.Println("\nMultiplexed Parallel Convolution (for complex AI model) time test started!")

	convIDs := []string{"CvTConv1", "CvTConv2", "MuseConv"}
	// convIDs := []string{"MuseConv"}
	//Set iter
	iter := 1

	minStartCipherLevel := 2
	maxStartCipherLevel := cc.Params.MaxLevel()
	// maxStartCipherLevel := 2

	for index := 0; index < len(convIDs); index++ {
		// Get ConvID
		convID := convIDs[index]
		if convID == "MuseConv" {
			cc = setCKKSEnvUseParamSet("PN15QP880CI")
			fmt.Println("CKKS parameter set as : PN15QP880CI")

			maxStartCipherLevel = cc.Params.MaxLevel()
		}

		//register index of rotation
		rots := modules.MulParConvRegister(convID)
		// fmt.Println(len(rots[0]), len(rots[1]), len(rots[2]), len(rots[0])+len(rots[1])+len(rots[2]))
		// continue

		//rotation key register
		newEvaluator := rotIndexToGaloisEl(int2dTo1d(rots), cc.Params, cc.Kgen, cc.Sk)

		//make mulParConv instance
		conv := modules.NewMulParConv(newEvaluator, cc.Encoder, cc.Params, convID)

		// Make input and kernel
		cf := conv.ConvFeature
		plainInput := makeRandomInput(cf.InputDataChannel, cf.InputDataHeight, cf.InputDataWidth)
		plainKernel := makeRandomKernel(cf.KernelNumber, cf.InputDataChannel, cf.KernelSize, cf.KernelSize)

		//Plaintext Convolution
		plainOutput := PlainConvolution2D(plainInput, plainKernel, cf.Stride, 1)
		// print3DArray(plainOutput)

		// Encrypt Input, Encode Kernel
		mulParPackedInput := MulParPacking(plainInput, cf, cc)
		conv.PreCompKernels = EncodeKernel(plainKernel, cf, cc)

		// Multiplexed parallel convolution Start!
		fmt.Printf("=== convID : %s, Depth : %v, CipherLevel : %v ~ %v, iter : %v === \n", convID, 2, minStartCipherLevel, maxStartCipherLevel, iter)
		fmt.Printf("startLevel executionTime(sec)\n")
		// MSE, RE, inf Norm
		var MSEList, REList, infNormList []float64
		for startCipherLevel := minStartCipherLevel; startCipherLevel <= maxStartCipherLevel; startCipherLevel++ {

			plain := ckks.NewPlaintext(cc.Params, startCipherLevel)
			cc.Encoder.Encode(mulParPackedInput, plain)
			inputCt, _ := cc.EncryptorSk.EncryptNew(plain)

			var totalForwardTime time.Duration
			//Conv Foward
			for i := 0; i < iter; i++ {
				//Convolution start
				start := time.Now()
				encryptedOutput := conv.Foward(inputCt)
				end := time.Now()
				totalForwardTime += end.Sub(start)

				// MSE, RE, inf Norm
				FHEOutput := UnMulParPacking(encryptedOutput, cf, cc)
				// print3DArray(FHEOutput)
				scores := MSE_RE_infNorm(plainOutput, FHEOutput)
				MSEList = append(MSEList, scores[0])
				REList = append(REList, scores[1])
				infNormList = append(infNormList, scores[2])
			}

			//Print Elapsed Time
			avgForwardTime := float64(totalForwardTime.Nanoseconds()) / float64(iter) / 1e9
			fmt.Printf("%v %v \n", startCipherLevel, avgForwardTime)
		}
		// Average Acc, Recall, F1 score
		MSEMin, MSEMax, MSEAvg := minMaxAvg(MSEList)
		REMin, REMax, REAvg := minMaxAvg(REList)
		infNormMin, infNormMax, infNormAvg := minMaxAvg(infNormList)

		fmt.Printf("MSE (Mean Squared Error)   : Min = %.2e, Max = %.2e, Avg = %.2e\n", MSEMin, MSEMax, MSEAvg)
		fmt.Printf("Relative Error             : Min = %.2e, Max = %.2e, Avg = %.2e\n", REMin, REMax, REAvg)
		fmt.Printf("Infinity Norm (L-infinity) : Min = %.2e, Max = %.2e, Avg = %.2e\n", infNormMin, infNormMax, infNormAvg)
		fmt.Println()
	}
}
func rotOptConvTimeTest(cc *customContext, depth int) {
	fmt.Printf("\nRotation Optimized Convolution (for %d-depth consumed) time test started!\n", depth)

	var convIDs []string

	switch depth {
	case 2:
		convIDs = []string{"CONV1", "CONV2", "CONV3s2", "CONV3", "CONV4s2", "CONV4"}
	case 3:
		convIDs = []string{"CONV2", "CONV3s2", "CONV3", "CONV4s2", "CONV4"}
	case 4:
		convIDs = []string{"CONV2", "CONV3s2", "CONV3", "CONV4s2", "CONV4"}
	case 5:
		convIDs = []string{"CONV3s2", "CONV4s2"}
	default:
		fmt.Printf("Unsupported depth: %d\n", depth)
		return
	}

	iter := 1
	minStartCipherLevel := depth
	maxStartCipherLevel := cc.Params.MaxLevel()
	// maxStartCipherLevel := depth

	for index := 0; index < len(convIDs); index++ {

		convID := convIDs[index]

		//register index of rotation
		rots := modules.RotOptConvRegister(convID, depth)

		//rotation key register
		newEvaluator := rotIndexToGaloisEl(int2dTo1d(rots), cc.Params, cc.Kgen, cc.Sk)

		//make rotOptConv instance
		conv := modules.NewrotOptConv(newEvaluator, cc.Encoder, cc.Params, convID, depth)

		// Make input and kernel
		cf := conv.ConvFeature
		plainInput := makeRandomInput(cf.InputDataChannel, cf.InputDataHeight, cf.InputDataWidth)
		plainKernel := makeRandomKernel(cf.KernelNumber, cf.InputDataChannel, cf.KernelSize, cf.KernelSize)

		//Plaintext Convolution
		plainOutput := PlainConvolution2D(plainInput, plainKernel, cf.Stride, 1)

		// Encrypt Input, Encode Kernel
		mulParPackedInput := MulParPacking(plainInput, cf, cc)
		conv.PreCompKernels = EncodeKernel(plainKernel, cf, cc)

		fmt.Printf("=== convID : %s, Depth : %v, CipherLevel : %v ~ %v, iter : %v === \n", convID, depth, Max(minStartCipherLevel, depth), maxStartCipherLevel, iter)
		fmt.Printf("startLevel executionTime(sec)\n")
		// MSE, RE, inf Norm
		var MSEList, REList, infNormList []float64
		for startCipherLevel := Max(minStartCipherLevel, depth); startCipherLevel <= maxStartCipherLevel; startCipherLevel++ {
			plain := ckks.NewPlaintext(cc.Params, startCipherLevel)
			cc.Encoder.Encode(mulParPackedInput, plain)
			inputCt, _ := cc.EncryptorSk.EncryptNew(plain)

			var totalForwardTime time.Duration
			//Conv Foward
			for i := 0; i < iter; i++ {
				//Convolution start
				start := time.Now()
				encryptedOutput := conv.Foward(inputCt)
				end := time.Now()
				totalForwardTime += end.Sub(start)

				// MSE, RE, inf Norm
				FHEOutput := UnMulParPacking(encryptedOutput, cf, cc)
				scores := MSE_RE_infNorm(plainOutput, FHEOutput)
				MSEList = append(MSEList, scores[0])
				REList = append(REList, scores[1])
				infNormList = append(infNormList, scores[2])
			}

			//Print Elapsed Time
			avgForwardTime := float64(totalForwardTime.Nanoseconds()) / float64(iter) / 1e9
			fmt.Printf("%v %v \n", startCipherLevel, avgForwardTime)
		}
		// Average Acc, Recall, F1 score
		MSEMin, MSEMax, MSEAvg := minMaxAvg(MSEList)
		REMin, REMax, REAvg := minMaxAvg(REList)
		infNormMin, infNormMax, infNormAvg := minMaxAvg(infNormList)

		fmt.Printf("MSE (Mean Squared Error)   : Min = %.2e, Max = %.2e, Avg = %.2e\n", MSEMin, MSEMax, MSEAvg)
		fmt.Printf("Relative Error             : Min = %.2e, Max = %.2e, Avg = %.2e\n", REMin, REMax, REAvg)
		fmt.Printf("Infinity Norm (L-infinity) : Min = %.2e, Max = %.2e, Avg = %.2e\n", infNormMin, infNormMax, infNormAvg)
		fmt.Println()
	}
}
func mulParConvTimeTest(cc *customContext) {
	fmt.Println("\nMultiplexed Parallel Convolution time test started!")

	convIDs := []string{"CONV1", "CONV2", "CONV3s2", "CONV3", "CONV4s2", "CONV4"}
	// convIDs := []string{"CONV4"}

	//Set iter
	iter := 1

	minStartCipherLevel := 2
	maxStartCipherLevel := cc.Params.MaxLevel()
	// maxStartCipherLevel := 2

	for index := 0; index < len(convIDs); index++ {
		convID := convIDs[index]

		//register index of rotation
		rots := modules.MulParConvRegister(convID)

		//rotation key register
		newEvaluator := rotIndexToGaloisEl(int2dTo1d(rots), cc.Params, cc.Kgen, cc.Sk)

		//make mulParConv instance
		conv := modules.NewMulParConv(newEvaluator, cc.Encoder, cc.Params, convID)

		// Make input and kernel
		cf := conv.ConvFeature
		plainInput := makeRandomInput(cf.InputDataChannel, cf.InputDataHeight, cf.InputDataWidth)
		plainKernel := makeRandomKernel(cf.KernelNumber, cf.InputDataChannel, cf.KernelSize, cf.KernelSize)

		//Plaintext Convolution
		plainOutput := PlainConvolution2D(plainInput, plainKernel, cf.Stride, 1)

		// Encrypt Input, Encode Kernel
		mulParPackedInput := MulParPacking(plainInput, cf, cc)
		conv.PreCompKernels = EncodeKernel(plainKernel, cf, cc)

		// Multiplexed parallel convolution Start!
		fmt.Printf("=== convID : %s, Depth : %v, CipherLevel : %v ~ %v, iter : %v === \n", convID, 2, minStartCipherLevel, maxStartCipherLevel, iter)
		fmt.Printf("startLevel executionTime(sec)\n")
		// MSE, RE, inf Norm
		var MSEList, REList, infNormList []float64
		for startCipherLevel := minStartCipherLevel; startCipherLevel <= maxStartCipherLevel; startCipherLevel++ {

			plain := ckks.NewPlaintext(cc.Params, startCipherLevel)
			cc.Encoder.Encode(mulParPackedInput, plain)
			inputCt, _ := cc.EncryptorSk.EncryptNew(plain)

			var totalForwardTime time.Duration
			//Conv Foward
			for i := 0; i < iter; i++ {
				//Convolution start
				start := time.Now()
				encryptedOutput := conv.Foward(inputCt)
				end := time.Now()
				totalForwardTime += end.Sub(start)

				// MSE, RE, inf Norm
				FHEOutput := UnMulParPacking(encryptedOutput, cf, cc)
				scores := MSE_RE_infNorm(plainOutput, FHEOutput)
				MSEList = append(MSEList, scores[0])
				REList = append(REList, scores[1])
				infNormList = append(infNormList, scores[2])
			}

			//Print Elapsed Time
			avgForwardTime := float64(totalForwardTime.Nanoseconds()) / float64(iter) / 1e9
			fmt.Printf("%v %v \n", startCipherLevel, avgForwardTime)
		}
		// Average Acc, Recall, F1 score
		MSEMin, MSEMax, MSEAvg := minMaxAvg(MSEList)
		REMin, REMax, REAvg := minMaxAvg(REList)
		infNormMin, infNormMax, infNormAvg := minMaxAvg(infNormList)

		fmt.Printf("MSE (Mean Squared Error)   : Min = %.2e, Max = %.2e, Avg = %.2e\n", MSEMin, MSEMax, MSEAvg)
		fmt.Printf("Relative Error             : Min = %.2e, Max = %.2e, Avg = %.2e\n", REMin, REMax, REAvg)
		fmt.Printf("Infinity Norm (L-infinity) : Min = %.2e, Max = %.2e, Avg = %.2e\n", infNormMin, infNormMax, infNormAvg)
		fmt.Println()
	}
}

func parBSGSfullyConnectedAccuracyTest(cc *customContext) {
	fmt.Println("Fully Connected + Parallel BSGS matrix-vector multiplication Test!")
	startLevel := 1
	endLevel := cc.Params.MaxLevel()
	// endLevel := 2
	//register
	rot := modules.ParBSGSFCRegister()

	//rot register
	newEvaluator := RotIndexToGaloisElements(rot, cc)

	//make avgPooling instance
	fc := modules.NewParBSGSFC(newEvaluator, cc.Encoder, cc.Params, 20)

	//Make input float data
	temp := txtToFloat("true_logs/AvgPoolEnd.txt")
	trueInputFloat := make([]float64, 32768)
	for i := 0; i < len(temp); i++ {
		for par := 0; par < 8; par++ {
			trueInputFloat[i+4096*par] = temp[i]
		}
	}

	//Make output float data
	trueOutputFloat := txtToFloat("true_logs/FcEnd.txt")

	var outputCt *rlwe.Ciphertext
	fmt.Printf("startLevel executionTime\n")
	for level := startLevel; level <= endLevel; level++ {
		// Encryption
		inputCt := floatToCiphertextLevel(trueInputFloat, level, cc.Params, cc.Encoder, cc.EncryptorSk)

		// Timer start
		startTime := time.Now()

		// AvgPooling Foward
		outputCt = fc.Foward(inputCt)

		// Timer end
		endTime := time.Now()

		// Print Elapsed Time
		// fmt.Printf("%v %v \n", level, TimeDurToFloatSec(endTime.Sub(startTime)))
		fmt.Printf("%v %v \n", level, (endTime.Sub(startTime)))

	}

	//Decryption
	outputFloat := ciphertextToFloat(outputCt, cc)

	scores := MSE_RE_infNorm_1D(outputFloat[0:10], trueOutputFloat)
	fmt.Printf("MSE (Mean Squared Error)   : %.2e\n", scores[0])
	fmt.Printf("Relative Error             : %.2e\n", scores[1])
	fmt.Printf("Infinity Norm (L-infinity) : %.2e\n", scores[2])
}

func mulParfullyConnectedAccuracyTest(cc *customContext) {
	fmt.Println("Fully Connected + Conventional BSGS diagonal matrix-vector multiplication Test!")
	startLevel := 1
	endLevel := cc.Params.MaxLevel()
	//register
	rot := modules.MulParFCRegister()

	//rot register
	newEvaluator := RotIndexToGaloisElements(rot, cc)

	//make avgPooling instance
	fc := modules.NewMulParFC(newEvaluator, cc.Encoder, cc.Params, 20)

	//Make input float data
	temp := txtToFloat("true_logs/AvgPoolEnd.txt")
	trueInputFloat := make([]float64, 32768)
	for i := 0; i < len(temp); i++ {
		for par := 0; par < 8; par++ {
			trueInputFloat[i+4096*par] = temp[i]
		}
	}

	//Make output float data
	trueOutputFloat := txtToFloat("true_logs/FcEnd.txt")

	var outputCt *rlwe.Ciphertext
	fmt.Printf("startLevel executionTime\n")
	for level := startLevel; level <= endLevel; level++ {
		// Encryption
		inputCt := floatToCiphertextLevel(trueInputFloat, level, cc.Params, cc.Encoder, cc.EncryptorSk)

		// Timer start
		startTime := time.Now()

		// AvgPooling Foward
		outputCt = fc.Foward(inputCt)

		// Timer end
		endTime := time.Now()

		// Print Elapsed Time
		// fmt.Printf("%v %v \n", level, TimeDurToFloatSec(endTime.Sub(startTime)))
		fmt.Printf("%v %v \n", level, (endTime.Sub(startTime)))

	}

	//Decryption
	outputFloat := ciphertextToFloat(outputCt, cc)

	scores := MSE_RE_infNorm_1D(outputFloat[0:10], trueOutputFloat)
	fmt.Printf("MSE (Mean Squared Error)   : %.2e\n", scores[0])
	fmt.Printf("Relative Error             : %.2e\n", scores[1])
	fmt.Printf("Infinity Norm (L-infinity) : %.2e\n", scores[2])
}

func RotIndexToGaloisElements(input []int, context *customContext) *ckks.Evaluator {
	var galElements []uint64

	for _, rotIndex := range input {
		galElements = append(galElements, context.Params.GaloisElement(rotIndex))
	}
	galKeys := context.Kgen.GenGaloisKeysNew(galElements, context.Sk)

	newEvaluator := ckks.NewEvaluator(context.Params, rlwe.NewMemEvaluationKeySet(context.Kgen.GenRelinearizationKeyNew(context.Sk), galKeys...))

	return newEvaluator
}

// Refer lattigo latest official document : https://pkg.go.dev/github.com/tuneinsight/lattigo/v4@v4.1.1/ckks#section-readme
func setCKKSEnvUseParamSet(paramSet string) *customContext {
	context := new(customContext)

	switch paramSet {
	case "PN16QP1761": // PN16QP1761 is a default parameter set for logN=16 and logQP = 1761
		context.Params, _ = ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
			LogN: 16,
			Q: []uint64{0x80000000080001, 0x2000000a0001, 0x2000000e0001, 0x1fffffc20001,
				0x200000440001, 0x200000500001, 0x200000620001, 0x1fffff980001,
				0x2000006a0001, 0x1fffff7e0001, 0x200000860001, 0x200000a60001,
				0x200000aa0001, 0x200000b20001, 0x200000c80001, 0x1fffff360001,
				0x200000e20001, 0x1fffff060001, 0x200000fe0001, 0x1ffffede0001,
				0x1ffffeca0001, 0x1ffffeb40001, 0x200001520001, 0x1ffffe760001,
				0x2000019a0001, 0x1ffffe640001, 0x200001a00001, 0x1ffffe520001,
				0x200001e80001, 0x1ffffe0c0001, 0x1ffffdee0001, 0x200002480001,
				0x1ffffdb60001, 0x200002560001},
			P:               []uint64{0x80000000440001, 0x7fffffffba0001, 0x80000000500001, 0x7fffffffaa0001},
			LogDefaultScale: 45,
		})
	case "PN15QP880CI": // PN16QP1761CI is a default parameter set for logN=16 and logQP = 1761
		context.Params, _ = ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
			LogN: 15,
			Q: []uint64{0x4000000120001,
				0x10000140001, 0xffffe80001, 0xffffc40001,
				0x100003e0001, 0xffffb20001, 0x10000500001,
				0xffff940001, 0xffff8a0001, 0xffff820001,
				0xffff780001, 0x10000960001, 0x10000a40001,
				0xffff580001, 0x10000b60001, 0xffff480001,
				0xffff420001, 0xffff340001},
			P:               []uint64{0x3ffffffd20001, 0x4000000420001, 0x3ffffffb80001},
			RingType:        ring.ConjugateInvariant,
			LogDefaultScale: 40,
		})
	case "PN16QP1654pq": // PN16QP1654pq is a default (post quantum) parameter set for logN=16 and logQP=1654
		context.Params, _ = ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
			LogN: 16,
			Q: []uint64{0x80000000080001, 0x2000000a0001, 0x2000000e0001, 0x1fffffc20001, 0x200000440001,
				0x200000500001, 0x200000620001, 0x1fffff980001, 0x2000006a0001, 0x1fffff7e0001,
				0x200000860001, 0x200000a60001, 0x200000aa0001, 0x200000b20001, 0x200000c80001,
				0x1fffff360001, 0x200000e20001, 0x1fffff060001, 0x200000fe0001, 0x1ffffede0001,
				0x1ffffeca0001, 0x1ffffeb40001, 0x200001520001, 0x1ffffe760001, 0x2000019a0001,
				0x1ffffe640001, 0x200001a00001, 0x1ffffe520001, 0x200001e80001, 0x1ffffe0c0001,
				0x1ffffdee0001, 0x200002480001},
			P:               []uint64{0x7fffffffe0001, 0x80000001c0001, 0x80000002c0001, 0x7ffffffd20001},
			LogDefaultScale: 45,
		})
	case "PN15QP827CIpq": // PN16QP1654CIpq is a default (post quantum) parameter set for logN=16 and logQP=1654
		context.Params, _ = ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
			LogN: 15,
			Q: []uint64{0x400000060001, 0x3fffe80001, 0x4000300001, 0x3fffb80001,
				0x40004a0001, 0x3fffb20001, 0x4000540001, 0x4000560001,
				0x3fff900001, 0x4000720001, 0x3fff8e0001, 0x4000800001,
				0x40008a0001, 0x3fff6c0001, 0x40009e0001, 0x3fff300001,
				0x3fff1c0001, 0x4000fc0001},
			P:               []uint64{0x2000000a0001, 0x2000000e0001, 0x1fffffc20001},
			RingType:        ring.ConjugateInvariant,
			LogDefaultScale: 38,
		})
	default:
		fmt.Printf("Unsupported CKKS parameter set name : %s\n", paramSet)
		fmt.Printf("CKKS setting set as default")
		return setCKKSEnv()
	}

	fmt.Printf("CKKS parameter set as : %s\n", paramSet)

	return setCKKSContext(context)
}

func setCKKSEnv() *customContext {
	context := new(customContext)
	// context.Params, _ = ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
	// 	LogN:            16,
	// 	LogQ:            []int{49, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40},
	// 	LogP:            []int{49, 49, 49},
	// 	LogDefaultScale: 40,
	// })

	context.Params, _ = ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN: 16,
		LogQ: []int{51,
			46, 46, 46, 46, 46,
			46, 46, 46, 46, 46,
			46, 46, 46, 46, 46,
			46, 46, 46, 46, 46,
			46, 46, 46, 46},
		LogP:            []int{60, 60, 60, 60, 60},
		LogDefaultScale: 46,
	})
	fmt.Printf("CKKS parameter set as : default\n")
	return setCKKSContext(context)
}

func setCKKSContext(context *customContext) *customContext {
	context.Kgen = ckks.NewKeyGenerator(context.Params)

	context.Sk, context.Pk = context.Kgen.GenKeyPairNew()

	context.Encoder = ckks.NewEncoder(context.Params)

	context.EncryptorPk = ckks.NewEncryptor(context.Params, context.Pk)

	context.EncryptorSk = ckks.NewEncryptor(context.Params, context.Sk)

	context.Decryptor = ckks.NewDecryptor(context.Params, context.Sk)

	galElements := []uint64{context.Params.GaloisElement(2)}
	galKeys := context.Kgen.GenGaloisKeysNew(galElements, context.Sk)

	context.Evaluator = ckks.NewEvaluator(context.Params, rlwe.NewMemEvaluationKeySet(context.Kgen.GenRelinearizationKeyNew(context.Sk), galKeys...))

	return context
}

func GalToEval(galKeys [][]*rlwe.GaloisKey, context *customContext) *ckks.Evaluator {
	var linGalKeys []*rlwe.GaloisKey

	for _, galKey := range galKeys {
		for _, g := range galKey {
			linGalKeys = append(linGalKeys, g)
		}
	}
	newEvaluator := ckks.NewEvaluator(context.Params, rlwe.NewMemEvaluationKeySet(context.Kgen.GenRelinearizationKeyNew(context.Sk), linGalKeys...))
	return newEvaluator
}

// Reorganize to -16384 ~ 16384. And remove repetitive elements.
func OrganizeRot(rotIndexes [][]int) [][]int {
	var result [][]int
	for level := 0; level < len(rotIndexes); level++ {
		//Reorganize
		rotateSets := make(map[int]bool)
		for _, each := range rotIndexes[level] {
			temp := each
			if temp > 16384 {
				temp = temp - 32768
			} else if temp < -16384 {
				temp = temp + 32768
			}
			rotateSets[temp] = true
		}
		//Change map to array
		var rotateArray []int
		for element := range rotateSets {
			if element != 0 {
				rotateArray = append(rotateArray, element)
			}
		}
		sort.Ints(rotateArray)
		//append to result
		result = append(result, rotateArray)
	}
	return result
}

// Extract current blueprint
func getBluePrint() {
	fmt.Println("Blue Print test started! Display all blueprint for convolution optimized convolutions.")
	fmt.Println("You can test other blue prints in modules/convConfig.go")

	convIDs := []string{"CONV1", "CONV2", "CONV3s2", "CONV3", "CONV4s2", "CONV4", "CvTConv1", "CvTConv2", "MuseConv"}
	maxDepth := []int{2, 4, 5, 4, 5, 4, 5, 6, 6}

	for index := 0; index < len(convIDs); index++ {
		for depth := 2; depth <= maxDepth[index]; depth++ {
			fmt.Printf("=== convID : %s, depth : %v === \n", convIDs[index], depth)
			convMap, _, _ := modules.GetConvBlueprints(convIDs[index], depth)
			rotSumBP := make([][]int, 1)
			rotSumBP[0] = []int{0}
			crossCombineBP := make([]int, 0)

			for d := 1; d < len(convMap); d++ {

				if convMap[d][0] == 3 {
					crossCombineBP = append(crossCombineBP, convMap[d][1])
					crossCombineBP = append(crossCombineBP, 0)
					crossCombineBP = append(crossCombineBP, convMap[d][2:]...)
					break
				} else {
					rotSumBP = append(rotSumBP, convMap[d])
				}

			}
			rotSumBP[0][0] = len(rotSumBP) - 1

			fmt.Println("RotationSumBP : ")
			fmt.Print("[")
			for _, row := range rotSumBP {
				fmt.Print("[")
				for i, val := range row {
					if i > 0 {
						fmt.Print(", ")
					}
					fmt.Printf("%d", val)
				}
				fmt.Print("],")
			}
			fmt.Println("]")

			fmt.Println("CrossCombineBP : ")
			fmt.Print("[")
			for i, val := range crossCombineBP {
				if i > 0 {
					fmt.Print(", ")
				}
				fmt.Printf("%d", val)
			}
			fmt.Println("]")

			fmt.Println("KernelBP : ")
			fmt.Print("[")
			for _, row := range modules.GetMulParConvFeature(convIDs[index]).KernelBP {
				fmt.Print("[")
				for i, val := range row {
					if i > 0 {
						fmt.Print(", ")
					}
					fmt.Printf("%d", val)
				}
				fmt.Print("],")
			}
			fmt.Println("]")
			fmt.Println()

		}

	}

}

func basicOperationTimeTest(cc *customContext) {
	floats := makeRandomFloat(32768)

	rot := make([]int, 1)
	rot[0] = 1

	fmt.Println("StartLevel Rotate Add Mul")
	//rot register
	newEvaluator := RotIndexToGaloisElements(rot, cc)
	for i := 0; i <= cc.Params.MaxLevel(); i++ {
		cipher1 := floatToCiphertextLevel(floats, i, cc.Params, cc.Encoder, cc.EncryptorSk)
		start1 := time.Now()
		newEvaluator.Rotate(cipher1, 1, cipher1)
		end1 := time.Now()

		start2 := time.Now()
		newEvaluator.Add(cipher1, cipher1, cipher1)
		end2 := time.Now()

		start3 := time.Now()
		newEvaluator.Mul(cipher1, cipher1, cipher1)
		end3 := time.Now()
		// newEvaluator.Rescale(cipher1, cipher1)
		// fmt.Println(i, TimeDurToFloatMiliSec(end1.Sub(start1)), TimeDurToFloatMiliSec(end2.Sub(start2)), TimeDurToFloatMiliSec(end3.Sub(start3)))
		fmt.Println(i, end1.Sub(start1), end2.Sub(start2), end3.Sub(start3))
	}

}

func bsgsMatVecMultAccuracyTest(N int, cc *customContext) {
	fmt.Println("\nConventional BSGS diagonal matrix-vector multiplication Test!")
	fmt.Println("matrix : ", N, "x", N, "  vector : ", N, "x", 1)
	nt := 32768

	fmt.Printf("=== Conventional (BSGS diag mat(N*N)-vec(N*1) mul) method start! N : %v ===\n", N)

	A := getMatrix(N, N)
	B := getMatrix(N, 1)

	//answer
	answer := originalMatMul(A, B)

	//change B to ciphertext
	B1d := make2dTo1d(B)
	B1d = resize(B1d, nt)
	//start mat vec mul
	rot := modules.BsgsDiagMatVecMulRegister(N)
	newEvaluator := RotIndexToGaloisElements(rot, cc)
	matVecMul := modules.NewBsgsDiagMatVecMul(A, N, nt, newEvaluator, cc.Encoder, cc.Params)

	fmt.Printf("startLevel executionTime\n")
	var MSEList, REList, infNormList []float64
	for level := 1; level <= cc.Params.MaxLevel(); level++ {

		Bct := floatToCiphertextLevel(B1d, level, cc.Params, cc.Encoder, cc.EncryptorSk)

		startTime := time.Now()
		BctOut := matVecMul.Foward(Bct)
		endTime := time.Now()
		outputFloat := ciphertextToFloat(BctOut, cc)

		scores := MSE_RE_infNorm_1D(outputFloat[0:N], make2dTo1d(answer))
		MSEList = append(MSEList, scores[0])
		REList = append(REList, scores[1])
		infNormList = append(infNormList, scores[2])

		fmt.Println(level, endTime.Sub(startTime))
	}
	MSEMin, MSEMax, MSEAvg := minMaxAvg(MSEList)
	REMin, REMax, REAvg := minMaxAvg(REList)
	infNormMin, infNormMax, infNormAvg := minMaxAvg(infNormList)

	fmt.Printf("MSE (Mean Squared Error)   : Min = %.2e, Max = %.2e, Avg = %.2e\n", MSEMin, MSEMax, MSEAvg)
	fmt.Printf("Relative Error             : Min = %.2e, Max = %.2e, Avg = %.2e\n", REMin, REMax, REAvg)
	fmt.Printf("Infinity Norm (L-infinity) : Min = %.2e, Max = %.2e, Avg = %.2e\n", infNormMin, infNormMax, infNormAvg)
}
func parBsgsMatVecMultAccuracyTest(N int, cc *customContext) {
	fmt.Println("\nParallel BSGS matrix-vector multiplication Test!")
	fmt.Println("matrix : ", N, "x", N, "  vector : ", N, "x", 1)
	nt := cc.Params.MaxSlots()

	pi := 1 //initially setting. (how many identical datas are in single ciphertext)

	fmt.Printf("=== Proposed (Parallely BSGS diag mat(N*N)-vec(N*1) mul) method start! N : %v ===\n", N)

	A := getMatrix(N, N)
	B := getMatrix(N, 1)

	answer := originalMatMul(A, B)

	B1d := make2dTo1d(B)
	B1d = resize(B1d, nt)
	for i := 1; i < pi; i *= 2 {
		tempB := rotate(B1d, -(nt/pi)*i)
		B1d = add(tempB, B1d)
	}
	//start mat vec mul
	rot := modules.ParBsgsDiagMatVecMulRegister(N, nt, pi)
	newEvaluator := RotIndexToGaloisElements(rot, cc)
	matVecMul := modules.NewParBsgsDiagMatVecMul(A, N, nt, pi, newEvaluator, cc.Encoder, cc.Params)

	fmt.Printf("startLevel executionTime\n")
	var MSEList, REList, infNormList []float64
	for level := 1; level <= cc.Params.MaxLevel(); level++ {

		Bct := floatToCiphertextLevel(B1d, level, cc.Params, cc.Encoder, cc.EncryptorSk)

		startTime := time.Now()
		BctOut := matVecMul.Foward(Bct)
		endTime := time.Now()
		outputFloat := ciphertextToFloat(BctOut, cc)

		scores := MSE_RE_infNorm_1D(outputFloat[0:N], make2dTo1d(answer))
		MSEList = append(MSEList, scores[0])
		REList = append(REList, scores[1])
		infNormList = append(infNormList, scores[2])

		fmt.Println(level, endTime.Sub(startTime))
	}
	MSEMin, MSEMax, MSEAvg := minMaxAvg(MSEList)
	REMin, REMax, REAvg := minMaxAvg(REList)
	infNormMin, infNormMax, infNormAvg := minMaxAvg(infNormList)

	fmt.Printf("MSE (Mean Squared Error)   : Min = %.2e, Max = %.2e, Avg = %.2e\n", MSEMin, MSEMax, MSEAvg)
	fmt.Printf("Relative Error             : Min = %.2e, Max = %.2e, Avg = %.2e\n", REMin, REMax, REAvg)
	fmt.Printf("Infinity Norm (L-infinity) : Min = %.2e, Max = %.2e, Avg = %.2e\n", infNormMin, infNormMax, infNormAvg)
}
func Contains(slice []string, str string) bool {
	for _, v := range slice {
		if v == str {
			return true
		}
	}
	return false
}

func floatToCiphertextLevel(floatInput []float64, level int, params ckks.Parameters, encoder *ckks.Encoder, encryptor *rlwe.Encryptor) *rlwe.Ciphertext {

	// encode to Plaintext
	exPlain := ckks.NewPlaintext(params, level)
	_ = encoder.Encode(floatInput, exPlain)

	// Encrypt to Ciphertext
	exCipher, err := encryptor.EncryptNew(exPlain)
	if err != nil {
		fmt.Println(err)
	}

	return exCipher
}

func makeGalois(cc *customContext, rotIndexes [][]int) [][]*rlwe.GaloisKey {

	galEls := make([][]*rlwe.GaloisKey, len(rotIndexes))

	for level := 0; level < len(rotIndexes); level++ {
		var galElements []uint64
		for _, rot := range rotIndexes[level] {
			galElements = append(galElements, cc.Params.GaloisElement(rot))
		}
		galKeys := cc.Kgen.GenGaloisKeysNew(galElements, cc.Sk)

		galEls = append(galEls, galKeys)

		fmt.Println(unsafe.Sizeof(*galKeys[0]), unsafe.Sizeof(galKeys[0].GaloisElement), unsafe.Sizeof(galKeys[0].NthRoot), unsafe.Sizeof(galKeys[0].EvaluationKey), unsafe.Sizeof(galKeys[0].GadgetCiphertext), unsafe.Sizeof(galKeys[0].BaseTwoDecomposition), unsafe.Sizeof(galKeys[0].Value))
	}
	// newEvaluator := ckks.NewEvaluator(cc.Params, rlwe.NewMemEvaluationKeySet(cc.Kgen.GenRelinearizationKeyNew(cc.Sk), galKeys...))
	return galEls
}

func rotIndexToGaloisEl(input []int, params ckks.Parameters, kgen *rlwe.KeyGenerator, sk *rlwe.SecretKey) *ckks.Evaluator {
	var galElements []uint64

	for _, rotIndex := range input {
		galElements = append(galElements, params.GaloisElement(rotIndex))
	}
	galKeys := kgen.GenGaloisKeysNew(galElements, sk)

	newEvaluator := ckks.NewEvaluator(params, rlwe.NewMemEvaluationKeySet(kgen.GenRelinearizationKeyNew(sk), galKeys...))

	return newEvaluator
}

func conv_time_loader() map[string]map[string]map[string]map[int]float64 {
	// Load data
	filePath := "true_logs/"
	method_name := []string{"mpconv", "ro_mpconv"}
	conv_depths := map[string][]string{
		"mpconv":    {"depth2"},
		"ro_mpconv": {"depth2", "depth3", "depth4", "depth5"},
	}
	conv_name := []string{"CONV1", "CONV2", "CONV3s2", "CONV3", "CONV4s2", "CONV4"}

	result := make(map[string]map[string]map[string]map[int]float64)

	for _, method := range method_name {
		result[method] = make(map[string]map[string]map[int]float64)
		for _, depth := range conv_depths[method] {
			result[method][depth] = make(map[string]map[int]float64)
			for _, conv_name := range conv_name {
				if method == "ro_mpconv" {
					if depth != "depth2" && (conv_name == "CONV1") {
						continue
					}
					if depth == "depth5" && (conv_name == "CONV1" || conv_name == "CONV2" || conv_name == "CONV3" || conv_name == "CONV4") {
						continue
					}
				}

				data := txtToFloatDict(filePath + method + "/" + depth + "/" + conv_name + "_time.txt")

				// for k, v := range data {
				// 	fmt.Printf("%s %s %d %g\n", method, conv_name, k, v)
				// }

				if result[method][conv_name] == nil {
					result[method][depth][conv_name] = make(map[int]float64)
				}
				result[method][depth][conv_name] = data
			}
		}
	}
	// Print data in order
	// for k1, v1 := range result {
	// 	for k2, v2 := range v1 {
	// 		for k3, v3 := range v2 {
	// 			for k4, v4 := range v3 {
	// 				fmt.Printf("%s -> %s -> %s -> %d = %.2f\n", k1, k2, k3, k4, v4)
	// 			}
	// 		}
	// 	}
	// }

	return result
}

type layer struct {
	// The above code is written in the Go programming language. It appears to be declaring a variable or
	// constant named "OperationName" without specifying its type or value. The three pound signs "
	// The above code is written in the Go programming language. It appears to be declaring a variable or
	// constant named "OperationName" without specifying its type or value. The triple hash symbol "
	OperationName string
	ConsumeLevel  int
}
type stage struct {
	layer_name string
	layer_num  int
}

func MPCNNTimeTest(cc *customContext, conv_time_dict map[string]map[string]map[string]map[int]float64) {
	fmt.Printf("\nApplication: MPConv time test started!\n")
	method_name := []string{"mpconv", "ro_mpconv"}
	maxLevel := 30
	startLevel := 18
	layers := map[string][]layer{
		"conv1": {
			{"CONV1", 2},
			{"AppReLU", 14},
		},
		"conv2": {
			{"CONV2", 2},
			{"Boot", 14},
			{"AppReLU", 14},
			{"CONV2", 2},
			{"Boot", 14},
			{"AppReLU", 14},
		},
		"conv3s2": {
			{"CONV3s2", 2},
			{"Boot", 14},
			{"AppReLU", 14},
			{"CONV3", 2},
			{"Boot", 14},
			{"AppReLU", 14},
		},
		"conv3": {
			{"CONV3", 2},
			{"Boot", 14},
			{"AppReLU", 14},
			{"CONV3", 2},
			{"Boot", 14},
			{"AppReLU", 14},
		},
		"conv4s2": {
			{"CONV4s2", 2},
			{"Boot", 14},
			{"AppReLU", 14},
			{"CONV4", 2},
			{"Boot", 14},
			{"AppReLU", 14},
		},
		"conv4": {
			{"CONV4", 2},
			{"Boot", 14},
			{"AppReLU", 14},
			{"CONV4", 2},
			{"Boot", 14},
			{"AppReLU", 14},
		},
	}

	models := map[string][]stage{
		"ResNet20": {
			{"conv1", 1},
			{"conv2", 3},
			{"conv3s2", 1},
			{"conv3", 2},
			{"conv4s2", 1},
			{"conv4", 2},
		},
		"ResNet32": {
			{"conv1", 1},
			{"conv2", 5},
			{"conv3s2", 1},
			{"conv3", 4},
			{"conv4s2", 1},
			{"conv4", 4},
		},
		"ResNet44": {
			{"conv1", 1},
			{"conv2", 7},
			{"conv3s2", 1},
			{"conv3", 6},
			{"conv4s2", 1},
			{"conv4", 6},
		},
		"ResNet56": {
			{"conv1", 1},
			{"conv2", 9},
			{"conv3s2", 1},
			{"conv3", 8},
			{"conv4s2", 1},
			{"conv4", 8},
		},
		"ResNet110": {
			{"conv1", 1},
			{"conv2", 18},
			{"conv3s2", 1},
			{"conv3", 17},
			{"conv4s2", 1},
			{"conv4", 17},
		},
	}

	for modelName, model := range models {
		for _, method := range method_name {
			fmt.Printf("\n=== Model: %s Method: %s ===\n", modelName, method)
			curLevel := startLevel
			totalTime := 0.0
			for _, stage := range model {
				stageName := stage.layer_name
				stageNum := stage.layer_num
				fmt.Printf("\n--- Stage: %s x %d ---\n", stageName, stageNum)
				for _, layer := range layers[stageName] {
					operationName := layer.OperationName
					consumeLevel := layer.ConsumeLevel

					if operationName == "Boot" {
						if curLevel != 0 {
							fmt.Printf("Warning! Bootstrpping doesn't conduct on level 0. (level: %d)\n", curLevel)
						}
						fmt.Printf("[Operation] %s, Cur Level: %d, Consume Level: %d\n", operationName, curLevel, consumeLevel)
						curLevel = maxLevel - consumeLevel
					} else if operationName == "AppReLU" {
						fmt.Printf("[Operation] %s, Cur Level: %d, Consume Level: %d\n", operationName, curLevel, consumeLevel)
						curLevel = curLevel - consumeLevel
					} else {
						if method == "mpconv" {
							consumeLevel = 2
						}
						curTime := conv_time_dict[method]["depth"+fmt.Sprint(consumeLevel)][operationName][curLevel]
						if curTime == 0 {
							fmt.Printf("No data for Layer: %s at Level: %d\n", operationName, curLevel)
							return
						}
						fmt.Printf("[Operation] %s, Cur Level: %d, Consume Level: %d, Time: %.5f sec, Total Time: %.2f sec\n", operationName, curLevel, consumeLevel, curTime, totalTime)

						curLevel = curLevel - consumeLevel
						totalTime += curTime * float64(stageNum)
					}

					if curLevel < 0 {
						fmt.Printf("Level is negative! Abort!\n")
						return
					}
				}
			}
			fmt.Printf("\n*** Model: %s Method: %s Total Time: %.2f sec ***\n", modelName, method, totalTime)
		}
	}

}

func autoFHEConvTimeTest(cc *customContext, conv_time_dict map[string]map[string]map[string]map[int]float64) {
	fmt.Printf("\nApplication: AutoFHE conv time test started!\n")
	method_name := []string{"mpconv", "ro_mpconv"}
	maxLevel := 30
	bootConsumeLevel := 14
	startLevel := 18
	debugMode := true

	models := map[string][]layer{
		"ResNet20_boot5": {
			// CONV1 x 1
			{"CONV1", 2},
			{"EvoReLU", 6},
			// CONV2 x 6
			{"StartResidual", 0},
			{"CONV2", 3},
			{"EvoReLU", 7},
			{"CONV2", 2},
			{"EndResidual", 0},
			{"EvoReLU", 0},

			{"StartResidual", 0},
			{"CONV2", 3},
			{"EvoReLU", 5},
			{"CONV2", 2},
			{"EndResidual", 0},
			{"EvoReLU", 2},

			{"StartResidual", 0},
			{"CONV2", 2},
			{"EvoReLU", 2},
			{"CONV2", 2},
			{"EndResidual", 0},
			{"EvoReLU", 0},

			// CONV3s2 x 1
			{"StartResidual", 0},
			{"CONV3s2", 2},
			{"EvoReLU", 2},
			// CONV3 x 5
			{"CONV3", 2},
			{"EndResidual", 0},
			{"EvoReLU", 2},

			{"StartResidual", 0},
			{"CONV3", 4},
			{"EvoReLU", 5},
			{"CONV3", 2},
			{"EndResidual", 0},
			{"EvoReLU", 0},

			{"StartResidual", 0},
			{"CONV3", 3},
			{"EvoReLU", 0},
			{"CONV3", 2}, // can be canceled
			{"EndResidual", 0},
			{"EvoReLU", 4},

			// CONV4s2 x 1
			{"StartResidual", 0},
			{"CONV4s2", 2},
			{"EvoReLU", 2},
			// CONV4 x 5
			{"CONV4", 2},
			{"EndResidual", 0},
			{"EvoReLU", 0},

			{"StartResidual", 0},
			{"CONV4", 2},
			{"EvoReLU", 2},
			{"CONV4", 2},
			{"EndResidual", 0},
			{"EvoReLU", 2},

			{"StartResidual", 0},
			{"CONV4", 4},
			{"EvoReLU", 2},
			{"CONV4", 4},
			{"EndResidual", 0},
			{"EvoReLU", 2},

			// FCLayer
			{"FCLayer", 2},
		},
		"ResNet20_boot11": {
			// CONV1 x 1
			{"CONV1", 2},
			{"EvoReLU", 6},
			// CONV2 x 6
			{"StartResidual", 0},
			{"CONV2", 2},
			{"EvoReLU", 8},
			{"CONV2", 4},
			{"EndResidual", 0},
			{"EvoReLU", 6},

			{"StartResidual", 0},
			{"CONV2", 4},
			{"EvoReLU", 8},
			{"CONV2", 4},
			{"EndResidual", 0},
			{"EvoReLU", 8},

			{"StartResidual", 0},
			{"CONV2", 2},
			{"EvoReLU", 2},
			{"CONV2", 2},
			{"EndResidual", 0},
			{"EvoReLU", 0},

			// CONV3s2 x 1
			{"StartResidual", 0},
			{"CONV3s2", 2},
			{"EvoReLU", 6},
			// CONV3 x 5
			{"CONV3", 4},
			{"EndResidual", 0},
			{"EvoReLU", 7},

			{"StartResidual", 0},
			{"CONV3", 2},
			{"EvoReLU", 5},
			{"CONV3", 2},
			{"EndResidual", 0},
			{"EvoReLU", 0},

			{"StartResidual", 0},
			{"CONV3", 2},
			{"EvoReLU", 5},
			{"CONV3", 2},
			{"EndResidual", 0},
			{"EvoReLU", 7},

			// CONV4s2 x 1
			{"StartResidual", 0},
			{"CONV4s2", 5},
			{"EvoReLU", 8},
			// CONV4 x 5
			{"CONV4", 3},
			{"EndResidual", 0},
			{"EvoReLU", 8},

			{"StartResidual", 0},
			{"CONV4", 4},
			{"EvoReLU", 6},
			{"CONV4", 4},
			{"EndResidual", 0},
			{"EvoReLU", 9},

			{"StartResidual", 0},
			{"CONV4", 4},
			{"EvoReLU", 9},
			{"CONV4", 3},
			{"EndResidual", 0},
			{"EvoReLU", 2},

			// FCLayer
			{"FCLayer", 2},
		},
		"ResNet32_boot8": {
			// CONV1
			{"CONV1", 2},
			{"EvoReLU", 0},
			// CONV2
			{"StartResidual", 0},
			{"CONV2", 2},
			{"EvoReLU", 2},
			{"CONV2", 2},
			{"EndResidual", 0},
			{"EvoReLU", 2},

			{"StartResidual", 0},
			{"CONV2", 4},
			{"EvoReLU", 2},
			{"CONV2", 2},
			{"EndResidual", 0},
			{"EvoReLU", 0},

			{"StartResidual", 0},
			{"CONV2", 2},
			{"EvoReLU", 2},
			{"CONV2", 2},
			{"EndResidual", 0},
			{"EvoReLU", 0},

			{"StartResidual", 0},
			{"CONV2", 2},
			{"EvoReLU", 0},
			{"CONV2", 2}, // can be canceled
			{"EndResidual", 0},
			{"EvoReLU", 0},

			{"StartResidual", 0},
			{"CONV2", 2},
			{"EvoReLU", 2},
			{"CONV2", 2},
			{"EndResidual", 0},
			{"EvoReLU", 0},

			// CONV3s2
			{"StartResidual", 0},
			{"CONV3s2", 2},
			{"EvoReLU", 2},
			// CONV3
			{"CONV3", 2},
			{"EndResidual", 0},
			{"EvoReLU", 2},

			{"StartResidual", 0},
			{"CONV3", 2},
			{"EvoReLU", 2},
			{"CONV3", 2},
			{"EndResidual", 0},
			{"EvoReLU", 0},

			{"StartResidual", 0},
			{"CONV3", 2},
			{"EvoReLU", 2},
			{"CONV3", 2},
			{"EndResidual", 0},
			{"EvoReLU", 0},

			{"StartResidual", 0},
			{"CONV3", 2},
			{"EvoReLU", 0},
			{"CONV3", 2}, // can be canceled
			{"EndResidual", 0},
			{"EvoReLU", 0},

			{"StartResidual", 0},
			{"CONV3", 2},
			{"EvoReLU", 2},
			{"CONV3", 2},
			{"EndResidual", 0},
			{"EvoReLU", 0},

			// CONV4s2
			{"StartResidual", 0},
			{"CONV4s2", 2},
			{"EvoReLU", 2},
			// CONV4
			{"CONV4", 4},
			{"EndResidual", 0},
			{"EvoReLU", 0},

			{"StartResidual", 0},
			{"CONV4", 2},
			{"EvoReLU", 2},
			{"CONV4", 4},
			{"EndResidual", 0},
			{"EvoReLU", 2},

			{"StartResidual", 0},
			{"CONV4", 2},
			// {"EvoReLU", 0},
			// {"CONV4", 2}, // can be canceled
			{"EndResidual", 0},
			{"EvoReLU", 0},

			{"StartResidual", 0},
			{"CONV4", 2},
			{"EvoReLU", 2},
			{"CONV4", 2},
			{"EndResidual", 0},
			{"EvoReLU", 2},

			{"StartResidual", 0},
			{"CONV4", 2},
			// {"EvoReLU", 0},
			// {"CONV4", 2}, // can be canceled
			{"EndResidual", 0},
			{"EvoReLU", 2},

			// FCLayer
			{"FCLayer", 2},
		},
		"ResNet32_boot19": {
			// CONV1
			{"CONV1", 2},
			{"EvoReLU", 0},
			// CONV2
			{"StartResidual", 0},
			{"CONV2", 4},
			{"EvoReLU", 9},
			{"CONV2", 3},
			{"EndResidual", 0},
			{"EvoReLU", 10},

			{"StartResidual", 0},
			{"CONV2", 4},
			{"EvoReLU", 8},
			{"CONV2", 4},
			{"EndResidual", 0},
			{"EvoReLU", 0},

			{"StartResidual", 0},
			{"CONV2", 4},
			{"EvoReLU", 9},
			{"CONV2", 4},
			{"EndResidual", 0},
			{"EvoReLU", 12},

			{"StartResidual", 0},
			{"CONV2", 2},
			{"EvoReLU", 0},
			{"CONV2", 2}, // can be canceled
			{"EndResidual", 0},
			{"EvoReLU", 0},

			{"StartResidual", 0},
			{"CONV2", 4},
			{"EvoReLU", 5},
			{"CONV2", 4},
			{"EndResidual", 0},
			{"EvoReLU", 0},

			// CONV3s2
			{"StartResidual", 0},
			{"CONV3s2", 3},
			{"EvoReLU", 12},
			// CONV3
			{"CONV3", 2},
			{"EndResidual", 0},
			{"EvoReLU", 0},

			{"StartResidual", 0},
			{"CONV3", 2},
			{"EvoReLU", 8},
			{"CONV3", 4},
			{"EndResidual", 0},
			{"EvoReLU", 11},

			{"StartResidual", 0},
			{"CONV3", 4},
			{"EvoReLU", 10},
			{"CONV3", 4},
			{"EndResidual", 0},
			{"EvoReLU", 0},

			{"StartResidual", 0},
			{"CONV3", 2},
			{"EvoReLU", 2},
			{"CONV3", 2},
			{"EndResidual", 0},
			{"EvoReLU", 0},

			{"StartResidual", 0},
			{"CONV3", 4},
			{"EvoReLU", 6},
			{"CONV3", 4},
			{"EndResidual", 0},
			{"EvoReLU", 9},

			// CONV4s2
			{"StartResidual", 0},
			{"CONV4s2", 3},
			{"EvoReLU", 2},
			// CONV4
			{"CONV4", 2},
			{"EndResidual", 0},
			{"EvoReLU", 5},

			{"StartResidual", 0},
			{"CONV4", 4},
			{"EvoReLU", 5},
			{"CONV4", 2},
			{"EndResidual", 0},
			{"EvoReLU", 9},

			{"StartResidual", 0},
			{"CONV4", 4},
			{"EvoReLU", 8},
			{"CONV4", 4},
			{"EndResidual", 0},
			{"EvoReLU", 7},

			{"StartResidual", 0},
			{"CONV4", 4},
			{"EvoReLU", 9},
			{"CONV4", 4},
			{"EndResidual", 0},
			{"EvoReLU", 8},

			{"StartResidual", 0},
			{"CONV4", 4},
			{"EvoReLU", 7},
			{"CONV4", 4},
			{"EndResidual", 0},
			{"EvoReLU", 0},

			// FCLayer
			{"FCLayer", 2},
		},
		"ResNet44_boot8": {
			// CONV1
			{"CONV1", 2},
			{"EvoReLU", 0},
			// CONV2
			{"StartResidual", 0},
			{"CONV2", 2},
			{"EvoReLU", 2},
			{"CONV2", 2},
			{"EndResidual", 0},
			{"EvoReLU", 0},

			{"StartResidual", 0},
			{"CONV2", 2},
			{"EvoReLU", 2},
			{"CONV2", 2},
			{"EndResidual", 0},
			{"EvoReLU", 0},

			{"StartResidual", 0},
			{"CONV2", 2},
			// {"EvoReLU", 0},
			// {"CONV2", 2}, // can be canceled
			{"EndResidual", 0},
			{"EvoReLU", 0},

			{"StartResidual", 0},
			{"CONV2", 2},
			// {"EvoReLU", 0},
			// {"CONV2", 2}, // can be canceled
			{"EndResidual", 0},
			{"EvoReLU", 0},

			{"StartResidual", 0},
			{"CONV2", 2},
			// {"EvoReLU", 0},
			// {"CONV2", 2}, // can be canceled
			{"EndResidual", 0},
			{"EvoReLU", 2},

			{"StartResidual", 0},
			{"CONV2", 2},
			// {"EvoReLU", 0},
			// {"CONV2", 2}, // can be canceled
			{"EndResidual", 0},
			{"EvoReLU", 0},

			{"StartResidual", 0},
			{"CONV2", 2},
			{"EvoReLU", 2},
			{"CONV2", 2},
			{"EndResidual", 0},
			{"EvoReLU", 0},

			// CONV3s2
			{"StartResidual", 0},
			{"CONV3s2", 4},
			{"EvoReLU", 2},
			// CONV3
			{"CONV3", 4},
			{"EndResidual", 0},
			{"EvoReLU", 0},

			{"StartResidual", 0},
			{"CONV3", 2},
			// {"EvoReLU", 0},
			// {"CONV3", 2}, // can be canceled
			{"EndResidual", 0},
			{"EvoReLU", 0},

			{"StartResidual", 0},
			{"CONV3", 2},
			{"EvoReLU", 2},
			{"CONV3", 4},
			{"EndResidual", 0},
			{"EvoReLU", 0},

			{"StartResidual", 0},
			{"CONV3", 2},
			{"EvoReLU", 2},
			{"CONV3", 2},
			{"EndResidual", 0},
			{"EvoReLU", 0},

			{"StartResidual", 0},
			{"CONV3", 2},
			{"EvoReLU", 2},
			{"CONV3", 2},
			{"EndResidual", 0},
			{"EvoReLU", 0},

			{"StartResidual", 0},
			{"CONV3", 2},
			// {"EvoReLU", 0},
			// {"CONV3", 2}, // can be canceled
			{"EndResidual", 0},
			{"EvoReLU", 0},

			{"StartResidual", 0},
			{"CONV3", 2},
			{"EvoReLU", 2},
			{"CONV3", 2},
			{"EndResidual", 0},
			{"EvoReLU", 0},

			// CONV4s2
			{"StartResidual", 0},
			{"CONV4s2", 2},
			{"EvoReLU", 2},
			// CONV4
			{"CONV4", 2},
			{"EndResidual", 0},
			{"EvoReLU", 0},

			{"StartResidual", 0},
			{"CONV4", 2},
			// {"EvoReLU", 0},
			// {"CONV4", 2}, // can be canceled
			{"EndResidual", 0},
			{"EvoReLU", 2},

			{"StartResidual", 0},
			{"CONV4", 2},
			// {"EvoReLU", 0},
			// {"CONV4", 2}, // can be canceled
			{"EndResidual", 0},
			{"EvoReLU", 0},

			{"StartResidual", 0},
			{"CONV4", 2},
			{"EvoReLU", 2},
			{"CONV4", 2},
			{"EndResidual", 0},
			{"EvoReLU", 2},

			{"StartResidual", 0},
			{"CONV4", 2},
			{"EvoReLU", 2},
			{"CONV4", 2},
			{"EndResidual", 0},
			{"EvoReLU", 0},

			{"StartResidual", 0},
			{"CONV4", 4},
			{"EvoReLU", 2},
			{"CONV4", 4},
			{"EndResidual", 0},
			{"EvoReLU", 0},

			{"StartResidual", 0},
			{"CONV4", 2},
			// {"EvoReLU", 0},
			// {"CONV4", 2}, // can be canceled
			{"EndResidual", 0},
			{"EvoReLU", 2},

			// FCLayer
			{"FCLayer", 2},
		},
		"ResNet44_boot22": {
			// CONV1
			{"CONV1", 2},
			{"EvoReLU", 0},
			// CONV2
			{"StartResidual", 0},
			{"CONV2", 3},
			// {"EvoReLU", 0},
			// {"CONV2", 2}, // can be canceled
			{"EndResidual", 0},
			{"EvoReLU", 0},

			{"StartResidual", 0},
			{"CONV2", 3},
			// {"EvoReLU", 0},
			// {"CONV2", 2}, // can be canceled
			{"EndResidual", 0},
			{"EvoReLU", 0},

			{"StartResidual", 0},
			{"CONV2", 3},
			// {"EvoReLU", 0},
			// {"CONV2", 2}, // can be canceled
			{"EndResidual", 0},
			{"EvoReLU", 0},

			{"StartResidual", 0},
			{"CONV2", 3},
			// {"EvoReLU", 0},
			// {"CONV2", 2}, // can be canceled
			{"EndResidual", 0},
			{"EvoReLU", 0},

			{"StartResidual", 0},
			{"CONV2", 2},
			// {"EvoReLU", 0},
			// {"CONV2", 2}, // can be canceled
			{"EndResidual", 0},
			{"EvoReLU", 0},

			{"StartResidual", 0},
			{"CONV2", 2},
			{"EvoReLU", 6},
			{"CONV2", 4},
			{"EndResidual", 0},
			{"EvoReLU", 13},

			{"StartResidual", 0},
			{"CONV2", 3},
			{"EvoReLU", 9},
			{"CONV2", 4},
			{"EndResidual", 0},
			{"EvoReLU", 7},

			// CONV3s2
			{"StartResidual", 0},
			{"CONV3s2", 5},
			{"EvoReLU", 8},
			// CONV3
			{"CONV3", 4},
			{"EndResidual", 0},
			{"EvoReLU", 0},

			{"StartResidual", 0},
			{"CONV3", 4},
			{"EvoReLU", 8},
			{"CONV3", 4},
			{"EndResidual", 0},
			{"EvoReLU", 8},

			{"StartResidual", 0},
			{"CONV3", 2},
			{"EvoReLU", 0},
			{"CONV3", 2}, // can be canceled
			{"EndResidual", 0},
			{"EvoReLU", 0},

			{"StartResidual", 0},
			{"CONV3", 4},
			{"EvoReLU", 9},
			{"CONV3", 3},
			{"EndResidual", 0},
			{"EvoReLU", 0},

			{"StartResidual", 0},
			{"CONV3", 4},
			{"EvoReLU", 8},
			{"CONV3", 4},
			{"EndResidual", 0},
			{"EvoReLU", 0},

			{"StartResidual", 0},
			{"CONV3", 4},
			{"EvoReLU", 8},
			{"CONV3", 4},
			{"EndResidual", 0},
			{"EvoReLU", 11},

			{"StartResidual", 0},
			{"CONV3", 4},
			{"EvoReLU", 6},
			{"CONV3", 4},
			{"EndResidual", 0},
			{"EvoReLU", 0},

			// CONV4s2
			{"StartResidual", 0},
			{"CONV4s2", 2},
			{"EvoReLU", 3},
			// CONV4
			{"CONV4", 4},
			{"EndResidual", 0},
			{"EvoReLU", 0},

			{"StartResidual", 0},
			{"CONV4", 4},
			{"EvoReLU", 10},
			{"CONV4", 4},
			{"EndResidual", 0},
			{"EvoReLU", 7},

			{"StartResidual", 0},
			{"CONV4", 4},
			{"EvoReLU", 12},
			{"CONV4", 2},
			{"EndResidual", 0},
			{"EvoReLU", 0},

			{"StartResidual", 0},
			{"CONV4", 2},
			{"EvoReLU", 2},
			{"CONV4", 4},
			{"EndResidual", 0},
			{"EvoReLU", 11},

			{"StartResidual", 0},
			{"CONV4", 4},
			{"EvoReLU", 10},
			{"CONV4", 2},
			{"EndResidual", 0},
			{"EvoReLU", 0},

			{"StartResidual", 0},
			{"CONV4", 4},
			{"EvoReLU", 9},
			{"CONV4", 3},
			{"EndResidual", 0},
			{"EvoReLU", 0},

			{"StartResidual", 0},
			{"CONV4", 4},
			{"EvoReLU", 10},
			{"CONV4", 4},
			{"EndResidual", 0},
			{"EvoReLU", 14},

			// FCLayer
			{"FCLayer", 2},
		},
	}

	for modelName, model := range models {
		for _, method := range method_name {
			fmt.Printf("\n=== Model: %s Method: %s ===\n", modelName, method)
			curLevel := startLevel
			totalTime := 0.0
			bootNum := 0
			beforeResidualLevel := 0
			residualFlag := false
			for _, layer := range model {
				operationName := layer.OperationName
				consumeLevel := layer.ConsumeLevel

				// Residual
				if operationName == "StartResidual" {
					residualFlag = true
					continue
				} else if operationName == "EndResidual" {
					if debugMode {
						fmt.Printf("[Residual] Residual End. Level becomes min(curLevel: %d, beforeResidualLevel: %d) -> %d\n", curLevel, beforeResidualLevel, int(math.Min(float64(curLevel), float64(beforeResidualLevel))))
					}
					curLevel = int(math.Min(float64(curLevel), float64(beforeResidualLevel)))
					continue
				}

				// Check & Operate
				// Set mpconv consume level
				if method == "mpconv" && (operationName != "EvoReLU" && operationName != "StartResidual" && operationName != "EndResidual") {
					consumeLevel = 2
				}
				// Bootstrapping check
				if curLevel < consumeLevel || curLevel == 0 {
					if debugMode {
						fmt.Printf("[Boot] Level %d -> %d\n", curLevel, maxLevel-bootConsumeLevel)
					}
					curLevel = maxLevel - bootConsumeLevel
					bootNum += 1
				}

				// Operation
				if operationName == "EvoReLU" {
					if debugMode {
						fmt.Printf("[Operation] %s, Cur Level: %d, Consume Level: %d\n", operationName, curLevel, consumeLevel)
					}
					curLevel = curLevel - consumeLevel
				} else if operationName == "FCLayer" {
					if debugMode {
						fmt.Printf("[Operation] %s, Cur Level: %d, Consume Level: %d\n", operationName, curLevel, consumeLevel)
					}
					curLevel = curLevel - consumeLevel
				} else {
					if residualFlag {
						if debugMode {
							fmt.Printf("[Residual] Before StartResidual, Cur Level: %d\n", curLevel)
						}
						beforeResidualLevel = curLevel
						residualFlag = false
					}
					curTime := conv_time_dict[method]["depth"+fmt.Sprint(consumeLevel)][operationName][curLevel]
					if curTime == 0 {
						fmt.Printf("No data for Layer: %s at Level: %d\n", operationName, curLevel)
						return
					}
					if debugMode {
						fmt.Printf("[Operation] %s, Cur Level: %d, Consume Level: %d, Time: %.5f sec, Total Time: %.2f sec\n", operationName, curLevel, consumeLevel, curTime, totalTime)
					}
					curLevel = curLevel - consumeLevel
					totalTime += curTime
				}
			}
			fmt.Printf("\n*** Model: %s Method: %s BootNum: %d Total Time: %.2f sec ***\n", modelName, method, bootNum, totalTime)
		}
	}

}

func cryptoFaceConvTimeTest(cc *customContext, conv_time_dict map[string]map[string]map[string]map[int]float64) {
	fmt.Printf("\nApplication: CryptoFace conv time test started!\n")
	method_name := []string{"mpconv", "ro_mpconv"}
	maxLevel := 30
	startLevel := 20

	layer := map[string][]layer{
		"SinglePatch": {
			{"CONV1", 2},
			// block1
			{"HerPN", 1},
			{"CONV2", 2},
			{"HerPN", 1},
			{"CONV2", 2},
			// block2
			{"HerPN", 1},
			{"CONV3s2", 2},
			{"HerPN", 1},
			{"CONV3", 2},
			// block3
			{"HerPN", 1},
			{"CONV3", 2},
			{"HerPN", 1},
			{"CONV3", 2},
			// block4
			{"HerPN", 1},
			{"CONV4s2", 2},
			{"HerPN", 1},
			{"CONV4", 2},
			//block5
			{"HerPN", 1},
			{"CONV4", 2},
		},
	}
	MPConvSinglePatchTime := 0.0
	roMPConvSinglePatchTime := 0.0
	for modelName, model := range layer {
		for _, method := range method_name {
			fmt.Printf("\n=== Calculating %s latency. Method: %s ===\n", modelName, method)
			curLevel := startLevel
			totalTime := 0.0
			for _, stage := range model {
				operationName := stage.OperationName
				consumeLevel := stage.ConsumeLevel

				if method == "mpconv" {
					consumeLevel = 2
				}

				// Bootstrapping check
				if curLevel < consumeLevel || curLevel == 0 {
					fmt.Printf("[Boot] Level %d -> %d\n", curLevel, maxLevel-14)
					curLevel = maxLevel - 14
				}

				// Operation
				if operationName == "HerPN" {
					fmt.Printf("[Operation] %s, Cur Level: %d, Consume Level: %d\n", operationName, curLevel, consumeLevel)
					curLevel = curLevel - consumeLevel
				} else if operationName == "CONV3s2" || operationName == "CONV4s2" {
					// Extra stride 2 convolution needs for residual connection
					newConsumeLevel := 2
					if method == "ro_mpconv" {
						newConsumeLevel = 5
					}
					newCurTime := conv_time_dict[method]["depth"+fmt.Sprint(newConsumeLevel)][operationName][curLevel]
					// Same with other convolution
					curTime := conv_time_dict[method]["depth"+fmt.Sprint(consumeLevel)][operationName][curLevel]
					if curTime == 0 {
						fmt.Printf("No data for Layer: %s at Level: %d\n", operationName, curLevel)
						return
					}
					fmt.Printf("[Operation] %s, Cur Level: %d, Consume Level: %d, Time: %.5f sec, Total Time: %.2f sec\n", operationName, curLevel, consumeLevel, curTime, totalTime)
					curLevel = curLevel - consumeLevel
					totalTime += curTime + newCurTime // add extra time
					if curLevel < 0 {
						fmt.Printf("Level is negative! Abort!\n")
						return
					}

				} else {
					curTime := conv_time_dict[method]["depth"+fmt.Sprint(consumeLevel)][operationName][curLevel]
					if curTime == 0 {
						fmt.Printf("No data for Layer: %s at Level: %d\n", operationName, curLevel)
						return
					}
					fmt.Printf("[Operation] %s, Cur Level: %d, Consume Level: %d, Time: %.5f sec, Total Time: %.2f sec\n", operationName, curLevel, consumeLevel, curTime, totalTime)
					curLevel = curLevel - consumeLevel
					totalTime += curTime
					if curLevel < 0 {
						fmt.Printf("Level is negative! Abort!\n")
						return
					}
				}
			}
			if method == "mpconv" {
				MPConvSinglePatchTime = totalTime
			} else if method == "ro_mpconv" {
				roMPConvSinglePatchTime = totalTime
			}
			fmt.Printf("\nMethod: %s Total Time: %.2f sec\n", method, totalTime)
		}
	}

	models := []string{"CryptoFace4", "CryptoFace9", "CryptoFace16"}
	singlePatchRepeatNum := []int{4, 9, 16}

	for i := 0; i < len(models); i++ {
		for _, method := range method_name {
			fmt.Printf("\n=== Model: %s Method: %s ===\n", models[i], method)
			singlePatchTime := 0.0
			if method == "mpconv" {
				singlePatchTime = MPConvSinglePatchTime
			} else if method == "ro_mpconv" {
				singlePatchTime = roMPConvSinglePatchTime
			}
			fmt.Printf("Total latency: %.2f sec\n", float64(singlePatchRepeatNum[i])*singlePatchTime)
		}
	}

}

func aespaConvTimeTest(cc *customContext, conv_time_dict map[string]map[string]map[string]map[int]float64) {
	fmt.Printf("\nApplication: AESPA (HerPN) time test started!\n")
	method_name := []string{"mpconv", "ro_mpconv"}
	maxLevel := 30
	startLevel := 20
	layers := map[string][]layer{
		"boot": {
			{"Boot", 14},
		},
		"conv1": {
			{"CONV1", 2},
			{"HerPN", 2},
		},
		"conv2": {
			{"CONV2", 2},
			{"HerPN", 2},
			{"CONV2", 2},
			{"HerPN", 2},
		},
		"conv3s2": {
			{"CONV3s2", 2},
			{"HerPN", 2},
			{"CONV3", 2},
			{"HerPN", 2},
		},
		"conv3": {
			{"CONV3", 2},
			{"HerPN", 2},
			{"CONV3", 2},
			{"HerPN", 2},
		},
		"conv4s2": {
			{"CONV4s2", 2},
			{"HerPN", 2},
			{"CONV4", 2},
			{"HerPN", 2},
		},
		"conv4": {
			{"CONV4", 2},
			{"HerPN", 2},
			{"CONV4", 2},
			{"HerPN", 2},
		},
	}

	models := map[string][]stage{
		"ResNet20": {
			// block1
			{"conv1", 1},
			// block2
			{"conv2", 2},
			{"boot", 1},
			{"conv2", 1},
			{"conv3s2", 1},
			{"boot", 1},
			// block3
			{"conv3", 2},
			{"boot", 1},
			// block4
			{"conv4s2", 1},
			{"conv4", 1},
			{"boot", 1},
			{"conv4", 1},
		},
		"ResNet32": {
			// block1
			{"conv1", 1},
			// block2
			{"conv2", 2},
			{"boot", 1},
			{"conv2", 2},
			{"boot", 1},
			{"conv2", 1},
			{"conv3s2", 1},
			{"boot", 1},
			// block3
			{"conv3", 2},
			{"boot", 1},
			{"conv3", 2},
			{"boot", 1},
			// block4
			{"conv4s2", 1},
			{"conv4", 1},
			{"boot", 1},
			{"conv4", 2},
			{"boot", 1},
			{"conv4", 1},
		},
		"ResNet44": {
			// block1
			{"conv1", 1},
			// block2
			{"conv2", 2},
			{"boot", 1},
			{"conv2", 2},
			{"boot", 1},
			{"conv2", 2},
			{"boot", 1},
			{"conv2", 1},
			{"conv3s2", 1},
			{"boot", 1},
			// block3
			{"conv3", 2},
			{"boot", 1},
			{"conv3", 2},
			{"boot", 1},
			{"conv3", 2},
			{"boot", 1},
			// block4
			{"conv4s2", 1},
			{"conv4", 1},
			{"boot", 1},
			{"conv4", 2},
			{"boot", 1},
			{"conv4", 2},
			{"boot", 1},
			{"conv4", 1},
		},
	}

	for modelName, model := range models {
		for _, method := range method_name {
			fmt.Printf("\n=== Model: %s Method: %s ===\n", modelName, method)
			curLevel := startLevel
			totalTime := 0.0
			for _, stage := range model {
				stageName := stage.layer_name
				stageNum := stage.layer_num
				fmt.Printf("\n--- Stage: %s x %d ---\n", stageName, stageNum)
				for i := 0; i < stageNum; i++ {
					for _, layer := range layers[stageName] {
						operationName := layer.OperationName
						consumeLevel := layer.ConsumeLevel

						if operationName == "Boot" {
							if curLevel != 0 {
								fmt.Printf("Warning! Bootstrpping doesn't conduct on level 0. (level: %d)\n", curLevel)
							}
							fmt.Printf("[Boot] Level %d -> %d\n", curLevel, maxLevel-consumeLevel)
							curLevel = maxLevel - consumeLevel
						} else if operationName == "HerPN" {
							fmt.Printf("[Operation] %s, Cur Level: %d, Consume Level: %d\n", operationName, curLevel, consumeLevel)
							curLevel = curLevel - consumeLevel
						} else {
							if method == "mpconv" {
								consumeLevel = 2
							}
							curTime := conv_time_dict[method]["depth"+fmt.Sprint(consumeLevel)][operationName][curLevel]
							if curTime == 0 {
								fmt.Printf("No data for Layer: %s at Level: %d\n", operationName, curLevel)
								return
							}
							fmt.Printf("[Operation] %s, Cur Level: %d, Consume Level: %d, Time: %.5f sec, Total Time: %.2f sec\n", operationName, curLevel, consumeLevel, curTime, totalTime)

							curLevel = curLevel - consumeLevel
							totalTime += curTime
						}

						if curLevel < 0 {
							fmt.Printf("Level is negative! Abort!\n")
							return
						}
					}
				}
			}
			fmt.Printf("\n*** Model: %s Method: %s Total Time: %.2f sec ***\n", modelName, method, totalTime)
		}
	}

}
