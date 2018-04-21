"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var tfc = require("@tensorflow/tfjs-core");
var executor_1 = require("../../executor");
var creation_executor_1 = require("./creation_executor");
var test_helper_1 = require("./test_helper");
describe('creation', function () {
    var node;
    var input1 = [tfc.tensor1d([1, 2, 3])];
    var input2 = [tfc.scalar(1)];
    var context = new executor_1.ExecutionContext({});
    beforeEach(function () {
        node = {
            name: 'test',
            op: '',
            category: 'creation',
            inputNames: ['input1', 'input2'],
            inputs: [],
            params: { x: test_helper_1.createTensorAttr(0) },
            children: []
        };
    });
    describe('executeOp', function () {
        describe('fill', function () {
            it('should call tfc.fill', function () {
                spyOn(tfc, 'fill');
                node.op = 'fill';
                node.params['shape'] = test_helper_1.createNumericArrayAttrFromIndex(0);
                node.params['value'] = test_helper_1.createNumberAttrFromIndex(1);
                creation_executor_1.executeOp(node, { input1: input1, input2: input2 }, context);
                expect(tfc.fill).toHaveBeenCalledWith([1, 2, 3], 1);
            });
        });
        describe('linspace', function () {
            it('should call tfc.linspace', function () {
                spyOn(tfc, 'linspace');
                node.op = 'linspace';
                node.params['start'] = test_helper_1.createNumberAttrFromIndex(0);
                node.params['stop'] = test_helper_1.createNumberAttrFromIndex(1);
                node.params['num'] = test_helper_1.createNumberAttrFromIndex(2);
                node.inputNames = ['input', 'input2', 'input3'];
                var input = [tfc.scalar(0)];
                var input3 = [tfc.scalar(2)];
                creation_executor_1.executeOp(node, { input: input, input2: input2, input3: input3 }, context);
                expect(tfc.linspace).toHaveBeenCalledWith(0, 1, 2);
            });
        });
        describe('oneHot', function () {
            it('should call tfc.oneHot', function () {
                spyOn(tfc, 'oneHot');
                node.op = 'oneHot';
                node.params['indices'] = test_helper_1.createNumericArrayAttrFromIndex(0);
                node.params['depth'] = test_helper_1.createNumberAttrFromIndex(1);
                node.params['onValue'] = test_helper_1.createNumberAttrFromIndex(2);
                node.params['offValue'] = test_helper_1.createNumberAttrFromIndex(3);
                node.inputNames = ['input', 'input2', 'input3', 'input4'];
                var input = [tfc.tensor1d([0])];
                var input3 = [tfc.scalar(2)];
                var input4 = [tfc.scalar(3)];
                creation_executor_1.executeOp(node, { input: input, input2: input2, input3: input3, input4: input4 }, context);
                expect(tfc.oneHot).toHaveBeenCalledWith([0], 1, 2, 3);
            });
        });
        describe('ones', function () {
            it('should call tfc.ones', function () {
                spyOn(tfc, 'ones');
                node.op = 'ones';
                node.params['shape'] = test_helper_1.createNumericArrayAttrFromIndex(0);
                node.params['dtype'] = test_helper_1.createDtypeAttr('float32');
                creation_executor_1.executeOp(node, { input1: input1 }, context);
                expect(tfc.ones).toHaveBeenCalledWith([1, 2, 3], 'float32');
            });
        });
        describe('onesLike', function () {
            it('should call tfc.onesLike', function () {
                spyOn(tfc, 'onesLike');
                node.op = 'onesLike';
                creation_executor_1.executeOp(node, { input1: input1 }, context);
                expect(tfc.onesLike).toHaveBeenCalledWith(input1[0]);
            });
        });
        describe('range', function () {
            it('should call tfc.range', function () {
                spyOn(tfc, 'range');
                node.op = 'range';
                node.params['start'] = test_helper_1.createNumberAttrFromIndex(0);
                node.params['stop'] = test_helper_1.createNumberAttr(1);
                node.params['step'] = test_helper_1.createNumberAttr(2);
                node.params['dtype'] = test_helper_1.createDtypeAttr('float32');
                node.inputNames = ['input', 'input2', 'input3'];
                var input = [tfc.scalar(0)];
                var input3 = [tfc.scalar(2)];
                creation_executor_1.executeOp(node, { input: input, input2: input2, input3: input3 }, context);
                expect(tfc.range).toHaveBeenCalledWith(0, 1, 2, 'float32');
            });
        });
        describe('randomUniform', function () {
            it('should call tfc.randomUniform', function () {
                spyOn(tfc, 'randomUniform');
                node.op = 'randomUniform';
                node.params['shape'] = test_helper_1.createNumericArrayAttrFromIndex(0);
                node.inputNames = ['input1'];
                node.params['maxval'] = test_helper_1.createNumberAttr(1);
                node.params['minval'] = test_helper_1.createNumberAttr(0);
                node.params['dtype'] = test_helper_1.createDtypeAttr('float32');
                node.params['seed'] = test_helper_1.createNumberAttr(0);
                creation_executor_1.executeOp(node, { input1: input1 }, context);
                expect(tfc.randomUniform)
                    .toHaveBeenCalledWith([1, 2, 3], 0, 1, 'float32');
            });
        });
        describe('truncatedNormal', function () {
            it('should call tfc.truncatedNormal', function () {
                spyOn(tfc, 'truncatedNormal');
                node.op = 'truncatedNormal';
                node.params['shape'] = test_helper_1.createNumericArrayAttrFromIndex(0);
                node.inputNames = ['input1'];
                node.params['stdDev'] = test_helper_1.createNumberAttr(1);
                node.params['mean'] = test_helper_1.createNumberAttr(0);
                node.params['dtype'] = test_helper_1.createDtypeAttr('float32');
                node.params['seed'] = test_helper_1.createNumberAttr(0);
                creation_executor_1.executeOp(node, { input1: input1 }, context);
                expect(tfc.truncatedNormal)
                    .toHaveBeenCalledWith([1, 2, 3], 0, 1, 'float32', 0);
            });
        });
        describe('zeros', function () {
            it('should call tfc.zeros', function () {
                spyOn(tfc, 'zeros');
                node.op = 'zeros';
                node.params['shape'] = test_helper_1.createNumericArrayAttrFromIndex(0);
                node.params['dtype'] = test_helper_1.createDtypeAttr('float32');
                creation_executor_1.executeOp(node, { input1: input1 }, context);
                expect(tfc.zeros).toHaveBeenCalledWith([1, 2, 3], 'float32');
            });
        });
        describe('zerosLike', function () {
            it('should call tfc.zerosLike', function () {
                spyOn(tfc, 'zerosLike');
                node.op = 'zerosLike';
                creation_executor_1.executeOp(node, { input1: input1 }, context);
                expect(tfc.zerosLike).toHaveBeenCalledWith(input1[0]);
            });
        });
    });
});
//# sourceMappingURL=creation_executor_test.js.map