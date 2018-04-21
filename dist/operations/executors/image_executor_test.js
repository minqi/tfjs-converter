"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var tfc = require("@tensorflow/tfjs-core");
var executor_1 = require("../../executor");
var image_executor_1 = require("./image_executor");
var test_helper_1 = require("./test_helper");
describe('image', function () {
    var node;
    var input1 = [tfc.tensor1d([1])];
    var context = new executor_1.ExecutionContext({});
    beforeEach(function () {
        node = {
            name: 'input1',
            op: '',
            category: 'image',
            inputNames: ['input1'],
            inputs: [],
            params: {},
            children: []
        };
    });
    describe('executeOp', function () {
        describe('resizeBilinear', function () {
            it('should return input', function () {
                node.op = 'resizeBilinear';
                node.params['images'] = test_helper_1.createTensorAttr(0);
                node.params['size'] = test_helper_1.createNumericArrayAttr([1, 2]);
                node.params['alignCorners'] = test_helper_1.createBoolAttr(true);
                spyOn(tfc.image, 'resizeBilinear');
                image_executor_1.executeOp(node, { input1: input1 }, context);
                expect(tfc.image.resizeBilinear)
                    .toHaveBeenCalledWith(input1[0], [1, 2], true);
            });
        });
    });
});
//# sourceMappingURL=image_executor_test.js.map