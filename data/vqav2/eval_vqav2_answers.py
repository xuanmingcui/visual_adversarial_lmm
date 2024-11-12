# coding: utf-8

import sys
from vqa import VQA
from vqaEval import VQAEval
import matplotlib.pyplot as plt
import skimage.io as io
import json
import random
import os
import argparse

def evaluate(args):
    # set up file names and paths
    # versionType ='v2_' # this should be '' when using VQA v2.0 dataset
    # taskType    ='OpenEnded' # 'OpenEnded' only for v2.0. 'OpenEnded' or 'MultipleChoice' for v1.0
    # dataType    ='mscoco'  # 'mscoco' only for v1.0. 'mscoco' for real and 'abstract_v002' for abstract for v1.0. 
    # dataSubType ='val2014'
    # annFile     ='%s/Annotations/%s%s_%s_annotations.json'%(dataDir, versionType, dataType, dataSubType)
    # quesFile    ='%s/Questions/%s%s_%s_%s_questions.json'%(dataDir, versionType, taskType, dataType, dataSubType)
    # imgDir      ='%s/Images/%s/%s/' %(dataDir, dataType, dataSubType)

    fileTypes   = ['results', 'accuracy', 'evalQA', 'evalQuesType', 'evalAnsType'] 

    # An example result json file has been provided in './Results' folder.  

    # [resFile, accuracyFile, evalQAFile, evalQuesTypeFile, evalAnsTypeFile] = ['%s/Results/%s%s_%s_%s_%s_%s.json'%(dataDir, versionType, taskType, dataType, dataSubType, \
    # resultType, fileType) for fileType in fileTypes]  

    with open(args.resFile, 'r') as f:
        res = json.load(f)

    # create vqa object and vqaRes object
    vqa = VQA(args.annFile, args.quesFile, question_ids=[q['question_id'] for q in res])
    vqaRes = vqa.loadRes(args.resFile, args.quesFile)

    # create vqaEval object by taking vqa and vqaRes
    vqaEval = VQAEval(vqa, vqaRes, n=2)   #n is precision of accuracy (number of places after decimal), default is 2

    # evaluate results
    """
    If you have a list of question ids on which you would like to evaluate your results, pass it as a list to below function
    By default it uses all the question ids in annotation file
    """
    vqaEval.evaluate() 

    print("=" * 50)

    print("==> Configurations:")

    print('Annotation File: ', args.annFile)
    print('Question File: ', args.quesFile)
    print('Image Directory: ', args.imgDir)
    print('Overall Accuracy: ', vqaEval.accuracy['overall'])

    print("=" * 50)

    print("Detailed Accuracy By Answer Type:")
    print("\n")

    print ("==> Per Question Type Accuracy is the following:")
    for quesType in vqaEval.accuracy['perQuestionType']:
    	print("%s : %.02f" %(quesType, vqaEval.accuracy['perQuestionType'][quesType]))
    print("\n")

    print("==> Per Answer Type Accuracy is the following:")
    for ansType in vqaEval.accuracy['perAnswerType']:
    	print("%s : %.02f" %(ansType, vqaEval.accuracy['perAnswerType'][ansType]))
    print("\n")
    # demo how to use evalQA to retrieve low score result
    evals = [quesId for quesId in vqaEval.evalQA if vqaEval.evalQA[quesId]<35]   #35 is per question percentage accuracy
    print("==> Low accuracy questions (accuracy<35): %d\n")
    if len(evals) > 0:
        for id in evals:
            print('Question id: %s. Question type: %s' %(id, vqa.loadQA(id)[0]['question_type']))
            
    	# print('ground truth answers')
    	# randomEval = random.choice(evals)
    	# randomAnn = vqa.loadQA(randomEval)
    	# vqa.showQA(randomAnn)

    	# print('\n')
    	# print('generated answer (accuracy %.02f)'%(vqaEval.evalQA[randomEval]))
    	# ann = vqaRes.loadQA(randomEval)[0]
    	# print("Answer:   %s\n" %(ann['answer']))

    	# imgId = randomAnn[0]['image_id']
    	# imgFilename = 'COCO_' + 'val2014' + '_'+ str(imgId).zfill(12) + '.jpg'
    	# if os.path.isfile(args.imgDir + imgFilename):
    	# 	I = io.imread(args.imgDir + imgFilename)
    	# 	plt.imshow(I)
    	# 	plt.axis('off')
    	# 	plt.show()

    # plot accuracy for various question types
    plt.bar(range(len(vqaEval.accuracy['perQuestionType'])), vqaEval.accuracy['perQuestionType'].values(), align='center')
    plt.xticks(range(len(vqaEval.accuracy['perQuestionType'])), vqaEval.accuracy['perQuestionType'].keys(), rotation=60, fontsize=3)
    plt.title('Per Question Type Accuracy', fontsize=10)
    plt.xlabel('Question Types', fontsize=10)
    plt.ylabel('Accuracy', fontsize=10)
    plt.savefig(os.path.join(args.resultFolder, 'figure.png'), dpi=300)

    # save evaluation results to ./Results folder
    json.dump(vqaEval.accuracy,     open(os.path.join(args.resultFolder, 'accuracy.json'),'w'))
    json.dump(vqaEval.evalQA,       open(os.path.join(args.resultFolder, 'evalQA.json'),'w'))
    json.dump(vqaEval.evalQuesType, open(os.path.join(args.resultFolder, 'evalQuestype.json'),'w'))
    json.dump(vqaEval.evalAnsType,  open(os.path.join(args.resultFolder, 'evalAnsType.json'),'w'))

    with open(args.outputFile, 'a') as f:
        f.write('Annotation File: {}\n'.format(args.annFile))
        f.write('Question File: {}\n'.format(args.quesFile))
        f.write('Image Directory: {}\n'.format(args.imgDir))
        f.write('Results File: {}\n'.format(args.resFile))
        f.write('Overall Accuracy: {}\n'.format(vqaEval.accuracy['overall']))
        f.write("=" * 50 + '\n')

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--annFile", type=str, default='data/vqav2/coco2014val_annotations.json')
    parser.add_argument("--quesFile", type=str, required=False, default='data/vqav2/coco2014val_questions.json')
    parser.add_argument("--imgDir", type=str, required=False, default='data/coco/val2014')
    parser.add_argument("--resFile", type=str, required=True, default=None)
    parser.add_argument("--resultFolder", type=str, required=True, default=None)
    parser.add_argument("--outputFile", type=str, required=False, default='results/vqav2_results.txt')

    args = parser.parse_args()
    evaluate(args)