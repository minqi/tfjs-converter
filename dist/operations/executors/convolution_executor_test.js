"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var tfc = require("@tensorflow/tfjs-core");
var executor_1 = require("../../executor");
var convolution_executor_1 = require("./convolution_executor");
var test_helper_1 = require("./test_helper");
describe('convolution', function () {
    var node;
    var input = [tfc.scalar(1)];
    var context = new executor_1.ExecutionContext({});
    beforeEach(function () {
        node = {
            name: 'test',
            op: '',
            category: 'convolution',
            inputNames: ['input'],
            inputs: [],
            params: { x: test_helper_1.createTensorAttr(0) },
            children: []
        };
    });
    describe('executeOp', function () {
        describe('avgPool', function () {
            it('should call tfc.avgPool', function () {
                spyOn(tfc, 'avgPool');
                node.op = 'avgPool';
                node.params['strides'] = test_helper_1.createNumericArrayAttr([1, 2, 2, 1]);
                node.params['pad'] = test_helper_1.createStrAttr('same');
                node.params['kernelSize'] = test_helper_1.createNumericArrayAttr([1, 2, 2, 1]);
                convolution_executor_1.executeOp(node, { input: input }, context);
                expect(tfc.avgPool)
                    .toHaveBeenCalledWith(input[0], [2, 2], [2, 2], 'same');
            });
        });
        describe('maxPool', function () {
            it('should call tfc.maxPool', function () {
                spyOn(tfc, 'maxPool');
                node.op = 'maxPool';
                node.params['strides'] = test_helper_1.createNumericArrayAttr([1, 2, 2, 1]);
                node.params['pad'] = test_helper_1.createStrAttr('same');
                node.params['kernelSize'] = test_helper_1.createNumericArrayAttr([1, 2, 2, 1]);
                convolution_executor_1.executeOp(node, { input: input }, context);
                expect(tfc.maxPool)
                    .toHaveBeenCalledWith(input[0], [2, 2], [2, 2], 'same');
            });
        });
        describe('Conv2d', function () {
            it('should call tfc.conv2d', function () {
                spyOn(tfc, 'conv2d');
                node.op = 'conv2d';
                node.params['filter'] = test_helper_1.createTensorAttr(1);
                node.params['strides'] = test_helper_1.createNumericArrayAttr([1, 2, 2, 1]);
                node.params['pad'] = test_helper_1.createStrAttr('same');
                node.params['dataFormat'] = test_helper_1.createStrAttr('NHWC');
                node.params['dilations'] = test_helper_1.createNumericArrayAttr([2, 2]);
                var input1 = [tfc.scalar(1.0)];
                var input2 = [tfc.scalar(1.0)];
                node.inputNames = ['input1', 'input2'];
                convolution_executor_1.executeOp(node, { input1: input1, input2: input2 }, context);
                expect(tfc.conv2d)
                    .toHaveBeenCalledWith(input1[0], input2[0], [2, 2], 'same', 'NHWC', [2, 2]);
            });
        });
        describe('conv2dTranspose', function () {
            it('should call tfc.conv2dTranspose', function () {
                spyOn(tfc, 'conv2dTranspose');
                node.op = 'conv2dTranspose';
                node.params['outputShape'] = test_helper_1.createNumericArrayAttr([1, 2, 2, 2]);
                node.params['filter'] = test_helper_1.createTensorAttr(1);
                node.params['strides'] = test_helper_1.createNumericArrayAttr([1, 2, 2, 1]);
                node.params['pad'] = test_helper_1.createStrAttr('same');
                var input1 = [tfc.scalar(1.0)];
                var input2 = [tfc.scalar(1.0)];
                node.inputNames = ['input1', 'input2'];
                convolution_executor_1.executeOp(node, { input1: input1, input2: input2 }, context);
                expect(tfc.conv2dTranspose)
                    .toHaveBeenCalledWith(input1[0], input2[0], [1, 2, 2, 2], [2, 2], 'same');
            });
        });
        describe('Conv1d', function () {
            it('should call tfc.conv1d', function () {
                spyOn(tfc, 'conv1d');
                node.op = 'conv1d';
                node.category = 'convolution';
                node.params['filter'] = test_helper_1.createTensorAttr(1);
                node.params['stride'] = test_helper_1.createNumberAttr(1);
                node.params['pad'] = test_helper_1.createStrAttr('same');
                node.params['dataFormat'] = test_helper_1.createStrAttr('NWC');
                node.params['dilation'] = test_helper_1.createNumberAttr(1);
                var input1 = [tfc.scalar(1.0)];
                var input2 = [tfc.scalar(1.0)];
                node.inputNames = ['input1', 'input2'];
                convolution_executor_1.executeOp(node, { input1: input1, input2: input2 }, context);
                expect(tfc.conv1d)
                    .toHaveBeenCalledWith(input1[0], input2[0], 1, 'same', 'NWC', 1);
            });
        });
        describe('depthwiseConv2d', function () {
            it('should call tfc.depthwiseConv2d', function () {
                spyOn(tfc, 'depthwiseConv2d');
                node.op = 'depthwiseConv2d';
                node.category = 'convolution';
                node.params['input'] = test_helper_1.createTensorAttr(0);
                node.params['filter'] = test_helper_1.createTensorAttr(1);
                node.params['strides'] = test_helper_1.createNumericArrayAttr([1, 2, 2, 1]);
                node.params['pad'] = test_helper_1.createStrAttr('same');
                node.params['dataFormat'] = test_helper_1.createStrAttr('NHWC');
                node.params['dilations'] = test_helper_1.createNumericArrayAttr([2, 2]);
                var input1 = [tfc.scalar(1.0)];
                var input2 = [tfc.scalar(1.0)];
                node.inputNames = ['input1', 'input2'];
                convolution_executor_1.executeOp(node, { input1: input1, input2: input2 }, context);
                expect(tfc.depthwiseConv2d)
                    .toHaveBeenCalledWith(input1[0], input2[0], [2, 2], 'same', 'NHWC', [2, 2]);
            });
        });
    });
});
//# sourceMappingURL=convolution_executor_test.js.map