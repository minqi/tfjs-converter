"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var tfc = require("@tensorflow/tfjs-core");
var executor_1 = require("../../executor");
var slice_join_executor_1 = require("./slice_join_executor");
var test_helper_1 = require("./test_helper");
describe('slice join', function () {
    var node;
    var input1 = [tfc.scalar(1)];
    var input2 = [tfc.scalar(2)];
    var input3 = [tfc.scalar(3)];
    var context = new executor_1.ExecutionContext({});
    describe('multi-tensor ops', function () {
        beforeEach(function () {
            node = {
                name: 'test',
                op: '',
                category: 'slice_join',
                inputNames: ['input1', 'input2', 'input3'],
                inputs: [],
                params: {
                    tensors: test_helper_1.createTensorsAttr(0, 1),
                    axis: test_helper_1.createNumberAttrFromIndex(-1)
                },
                children: []
            };
        });
        describe('executeOp', function () {
            ['concat', 'stack'].forEach(function (op) {
                it('should call tfc.' + op, function () {
                    var spy = spyOn(tfc, op);
                    node.op = op;
                    slice_join_executor_1.executeOp(node, { input1: input1, input2: input2, input3: input3 }, context);
                    expect(spy).toHaveBeenCalledWith([input1[0], input2[0]], 3);
                });
            });
        });
    });
    describe('single-tensor ops', function () {
        beforeEach(function () {
            node = {
                name: 'test',
                op: '',
                category: 'slice_join',
                inputNames: ['input1'],
                inputs: [],
                params: { x: test_helper_1.createTensorAttr(0) },
                children: []
            };
        });
        describe('executeOp', function () {
            it('should call tfc.reverse', function () {
                spyOn(tfc, 'reverse');
                node.op = 'reverse';
                node.params.axis = test_helper_1.createNumberAttrFromIndex(1);
                node.inputNames = ['input1', 'input2'];
                slice_join_executor_1.executeOp(node, { input1: input1, input2: input2 }, context);
                expect(tfc.reverse).toHaveBeenCalledWith(input1[0], 2);
            });
            it('should call tfc.tile', function () {
                spyOn(tfc, 'tile');
                node.op = 'tile';
                node.params.reps = test_helper_1.createNumberAttrFromIndex(1);
                node.inputNames = ['input1', 'input2'];
                slice_join_executor_1.executeOp(node, { input1: input1, input2: input2 }, context);
                expect(tfc.tile).toHaveBeenCalledWith(input1[0], 2);
            });
            it('should call tfc.slice', function () {
                spyOn(tfc, 'slice');
                node.op = 'slice';
                node.params.begin = test_helper_1.createNumericArrayAttr([1]);
                node.params.size = test_helper_1.createNumericArrayAttr([2]);
                slice_join_executor_1.executeOp(node, { input1: input1 }, context);
                expect(tfc.slice).toHaveBeenCalledWith(input1[0], [1], [2]);
            });
            it('should call tfc.gather', function () {
                spyOn(tfc, 'gather');
                node.op = 'gather';
                node.params.axis = test_helper_1.createNumberAttr(1);
                node.params.indices = test_helper_1.createTensorAttr(1);
                node.inputNames = ['input1', 'input2'];
                slice_join_executor_1.executeOp(node, { input1: input1, input2: input2 }, context);
                expect(tfc.gather).toHaveBeenCalledWith(input1[0], input2[0], 1);
            });
        });
    });
});
//# sourceMappingURL=slice_join_executor_test.js.map