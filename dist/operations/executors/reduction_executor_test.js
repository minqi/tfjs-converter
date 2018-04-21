"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var tfc = require("@tensorflow/tfjs-core");
var executor_1 = require("../../executor");
var reduction_executor_1 = require("./reduction_executor");
var test_helper_1 = require("./test_helper");
describe('reduction', function () {
    var node;
    var input1 = [tfc.scalar(1)];
    var context = new executor_1.ExecutionContext({});
    beforeEach(function () {
        node = {
            name: 'test',
            op: '',
            category: 'logical',
            inputNames: ['input1'],
            inputs: [],
            params: {
                x: test_helper_1.createTensorAttr(0),
                axis: test_helper_1.createNumberAttr(1),
                keepDims: test_helper_1.createBoolAttr(true)
            },
            children: []
        };
    });
    describe('executeOp', function () {
        ['max', 'mean', 'min', 'sum'].forEach(function (op) {
            it('should call tfc.' + op, function () {
                var spy = spyOn(tfc, op);
                node.op = op;
                reduction_executor_1.executeOp(node, { input1: input1 }, context);
                expect(spy).toHaveBeenCalledWith(input1[0], 1, true);
            });
        });
        describe('argMax', function () {
            it('should call tfc.argMax', function () {
                spyOn(tfc, 'argMax');
                node.op = 'argMax';
                reduction_executor_1.executeOp(node, { input1: input1 }, context);
                expect(tfc.argMax).toHaveBeenCalledWith(input1[0], 1);
            });
        });
        describe('argMin', function () {
            it('should call tfc.argMin', function () {
                spyOn(tfc, 'argMin');
                node.op = 'argMin';
                reduction_executor_1.executeOp(node, { input1: input1 }, context);
                expect(tfc.argMin).toHaveBeenCalledWith(input1[0], 1);
            });
        });
    });
});
//# sourceMappingURL=reduction_executor_test.js.map