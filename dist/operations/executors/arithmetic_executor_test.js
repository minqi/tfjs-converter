"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var tfc = require("@tensorflow/tfjs-core");
var executor_1 = require("../../executor");
var arithmetic_executor_1 = require("./arithmetic_executor");
var test_helper_1 = require("./test_helper");
describe('arithmetic', function () {
    var node;
    var input1 = [tfc.scalar(1)];
    var input2 = [tfc.scalar(1)];
    var context = new executor_1.ExecutionContext({});
    beforeEach(function () {
        node = {
            name: 'test',
            op: '',
            category: 'arithmetic',
            inputNames: ['input1', 'input2'],
            inputs: [],
            params: { a: test_helper_1.createTensorAttr(0), b: test_helper_1.createTensorAttr(1) },
            children: []
        };
    });
    describe('executeOp', function () {
        ['add', 'mul', 'div', 'sub', 'maximum', 'minimum', 'pow'].forEach((function (op) {
            it('should call tfc.' + op, function () {
                var spy = spyOn(tfc, op);
                node.op = op;
                arithmetic_executor_1.executeOp(node, { input1: input1, input2: input2 }, context);
                expect(spy).toHaveBeenCalledWith(input1[0], input2[0]);
            });
        }));
    });
});
//# sourceMappingURL=arithmetic_executor_test.js.map