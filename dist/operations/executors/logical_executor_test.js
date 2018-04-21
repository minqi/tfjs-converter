"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var tfc = require("@tensorflow/tfjs-core");
var executor_1 = require("../../executor");
var logical_executor_1 = require("./logical_executor");
var test_helper_1 = require("./test_helper");
describe('logical', function () {
    var node;
    var input1 = [tfc.scalar(1)];
    var input2 = [tfc.scalar(2)];
    var context = new executor_1.ExecutionContext({});
    beforeEach(function () {
        node = {
            name: 'test',
            op: '',
            category: 'logical',
            inputNames: ['input1', 'input2'],
            inputs: [],
            params: { a: test_helper_1.createTensorAttr(0), b: test_helper_1.createTensorAttr(1) },
            children: []
        };
    });
    describe('executeOp', function () {
        ['equal', 'greater', 'greaterEqual', 'less', 'lessEqual', 'logicalAnd',
            'logicalOr']
            .forEach(function (op) {
            it('should call tfc.' + op, function () {
                var spy = spyOn(tfc, op);
                node.op = op;
                logical_executor_1.executeOp(node, { input1: input1, input2: input2 }, context);
                expect(spy).toHaveBeenCalledWith(input1[0], input2[0]);
            });
        });
        describe('logicalNot', function () {
            it('should call tfc.logicalNot', function () {
                spyOn(tfc, 'logicalNot');
                node.op = 'logicalNot';
                logical_executor_1.executeOp(node, { input1: input1 }, context);
                expect(tfc.logicalNot).toHaveBeenCalledWith(input1[0]);
            });
        });
        describe('where', function () {
            it('should call tfc.where', function () {
                spyOn(tfc, 'where');
                node.op = 'where';
                node.inputNames = ['input1', 'input2', 'input3'];
                node.params.condition = test_helper_1.createTensorAttr(2);
                var input3 = [tfc.scalar(1)];
                logical_executor_1.executeOp(node, { input1: input1, input2: input2, input3: input3 }, context);
                expect(tfc.where).toHaveBeenCalledWith(input3[0], input1[0], input2[0]);
            });
        });
    });
});
//# sourceMappingURL=logical_executor_test.js.map