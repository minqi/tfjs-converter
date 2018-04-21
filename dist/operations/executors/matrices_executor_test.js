"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var tfc = require("@tensorflow/tfjs-core");
var executor_1 = require("../../executor");
var matrices_executor_1 = require("./matrices_executor");
var test_helper_1 = require("./test_helper");
describe('matrices', function () {
    var node;
    var input1 = [tfc.scalar(1)];
    var input2 = [tfc.scalar(2)];
    var context = new executor_1.ExecutionContext({});
    beforeEach(function () {
        node = {
            name: 'test',
            op: '',
            category: 'matrices',
            inputNames: ['input1', 'input2'],
            inputs: [],
            params: { a: test_helper_1.createTensorAttr(0), b: test_helper_1.createTensorAttr(1) },
            children: []
        };
    });
    describe('executeOp', function () {
        describe('matMul', function () {
            it('should call tfc.matMul', function () {
                spyOn(tfc, 'matMul');
                node.op = 'matMul';
                node.params.transposeA = test_helper_1.createBoolAttr(true);
                node.params.transposeB = test_helper_1.createBoolAttr(false);
                matrices_executor_1.executeOp(node, { input1: input1, input2: input2 }, context);
                expect(tfc.matMul)
                    .toHaveBeenCalledWith(input1[0], input2[0], true, false);
            });
        });
        describe('transpose', function () {
            it('should call tfc.transpose', function () {
                spyOn(tfc, 'transpose');
                node.op = 'transpose';
                node.inputNames = ['input1', 'input2', 'input3'];
                node.params = {
                    x: test_helper_1.createTensorAttr(0),
                    perm: test_helper_1.createNumericArrayAttr([1, 2])
                };
                matrices_executor_1.executeOp(node, { input1: input1 }, context);
                expect(tfc.transpose).toHaveBeenCalledWith(input1[0], [1, 2]);
            });
        });
    });
});
//# sourceMappingURL=matrices_executor_test.js.map