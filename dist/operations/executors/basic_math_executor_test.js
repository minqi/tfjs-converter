"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var tfc = require("@tensorflow/tfjs-core");
var executor_1 = require("../../executor");
var basic_math_executor_1 = require("./basic_math_executor");
var test_helper_1 = require("./test_helper");
describe('basic math', function () {
    var node;
    var input1 = [tfc.scalar(1)];
    var context = new executor_1.ExecutionContext({});
    beforeEach(function () {
        node = {
            name: 'test',
            op: '',
            category: 'basic_math',
            inputNames: ['input1'],
            inputs: [],
            params: { x: test_helper_1.createTensorAttr(0) },
            children: []
        };
    });
    describe('executeOp', function () {
        ['abs', 'acos', 'asin', 'atan', 'ceil', 'cos', 'cosh', 'elu', 'exp',
            'floor', 'log', 'relu', 'selu', 'sigmoid', 'sin', 'sinh', 'sqrt', 'square',
            'tanh', 'tan']
            .forEach(function (op) {
            it('should call tfc.' + op, function () {
                var spy = spyOn(tfc, op);
                node.op = op;
                basic_math_executor_1.executeOp(node, { input1: input1 }, context);
                expect(spy).toHaveBeenCalledWith(input1[0]);
            });
        });
        describe('clipByValue', function () {
            it('should call tfc.clipByValue', function () {
                spyOn(tfc, 'clipByValue');
                node.op = 'clipByValue';
                node.params['clipValueMax'] = test_helper_1.createNumberAttr(6);
                node.params['clipValueMin'] = test_helper_1.createNumberAttr(0);
                basic_math_executor_1.executeOp(node, { input1: input1 }, context);
                expect(tfc.clipByValue).toHaveBeenCalledWith(input1[0], 0, 6);
            });
        });
        describe('rsqrt', function () {
            it('should call tfc.div', function () {
                var input1 = [tfc.scalar(1)];
                node.op = 'rsqrt';
                spyOn(tfc, 'div');
                spyOn(tfc, 'sqrt').and.returnValue(input1);
                basic_math_executor_1.executeOp(node, { input1: input1 }, context);
                expect(tfc.sqrt).toHaveBeenCalledWith(input1[0]);
                expect(tfc.div).toHaveBeenCalledWith(jasmine.any(tfc.Tensor), input1);
            });
        });
    });
});
//# sourceMappingURL=basic_math_executor_test.js.map