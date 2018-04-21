"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var tfc = require("@tensorflow/tfjs-core");
var executor_1 = require("../../executor");
var graph_executor_1 = require("./graph_executor");
var test_helper_1 = require("./test_helper");
describe('graph', function () {
    var node;
    var input1 = [tfc.tensor1d([1])];
    var input2 = [tfc.tensor1d([1])];
    var context = new executor_1.ExecutionContext({});
    beforeEach(function () {
        node = {
            name: 'input1',
            op: '',
            category: 'graph',
            inputNames: [],
            inputs: [],
            params: {},
            children: []
        };
    });
    describe('executeOp', function () {
        describe('const', function () {
            it('should return input', function () {
                node.op = 'const';
                expect(graph_executor_1.executeOp(node, { input1: input1 }, context)).toEqual(input1);
            });
        });
        describe('placeholder', function () {
            it('should return input', function () {
                node.op = 'placeholder';
                expect(graph_executor_1.executeOp(node, { input1: input1 }, context)).toEqual(input1);
            });
            it('should return default if input not set', function () {
                node.inputNames = ['input2'];
                node.op = 'placeholder';
                node.params.default = test_helper_1.createTensorAttr(0);
                expect(graph_executor_1.executeOp(node, { input2: input2 }, context)).toEqual(input2);
            });
        });
        describe('identity', function () {
            it('should return input', function () {
                node.inputNames = ['input'];
                node.params.x = test_helper_1.createTensorAttr(0);
                node.op = 'identity';
                expect(graph_executor_1.executeOp(node, { input: input1 }, context)).toEqual(input1);
            });
        });
        describe('shape', function () {
            it('should return shape', function () {
                node.inputNames = ['input'];
                node.params.x = test_helper_1.createTensorAttr(0);
                node.op = 'shape';
                expect(Array.prototype.slice.call(graph_executor_1.executeOp(node, { input: input1 }, context)[0]
                    .dataSync()))
                    .toEqual([1]);
            });
        });
        describe('noop', function () {
            it('should return empty', function () {
                node.op = 'noop';
                expect(graph_executor_1.executeOp(node, {}, context)).toEqual([]);
            });
        });
    });
    describe('print', function () {
        it('should return empty', function () {
            node.op = 'print';
            node.inputNames = ['input1', 'input2'];
            node.params.x = test_helper_1.createTensorAttr(0);
            node.params.data = test_helper_1.createTensorsAttr(1, 1);
            node.params.message = test_helper_1.createStrAttr('message');
            node.params.summarize = test_helper_1.createNumberAttr(1);
            spyOn(console, 'log');
            spyOn(console, 'warn');
            expect(graph_executor_1.executeOp(node, { input1: input1, input2: input2 }, context)).toEqual(input1);
            expect(console.warn).toHaveBeenCalled();
            expect(console.log).toHaveBeenCalledWith('message');
            expect(console.log).toHaveBeenCalledWith([1]);
        });
    });
});
//# sourceMappingURL=graph_executor_test.js.map