"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var executor_1 = require("../executor");
var arithmetic = require("./executors/arithmetic_executor");
var basic_math = require("./executors/basic_math_executor");
var convolution = require("./executors/convolution_executor");
var creation = require("./executors/creation_executor");
var graph = require("./executors/graph_executor");
var image = require("./executors/image_executor");
var logical = require("./executors/logical_executor");
var matrices = require("./executors/matrices_executor");
var normalization = require("./executors/normalization_executor");
var reduction = require("./executors/reduction_executor");
var slice_join = require("./executors/slice_join_executor");
var transformation = require("./executors/transformation_executor");
var operation_executor_1 = require("./operation_executor");
describe('OperationExecutor', function () {
    var node;
    var context = new executor_1.ExecutionContext({});
    beforeEach(function () {
        node = {
            name: 'test',
            op: 'const',
            category: 'graph',
            inputNames: [],
            inputs: [],
            params: {},
            children: []
        };
    });
    describe('executeOp', function () {
        [arithmetic, basic_math, convolution, creation, image, graph, logical,
            matrices, normalization, reduction, slice_join, transformation]
            .forEach(function (category) {
            it('should call ' + category.CATEGORY + ' executor', function () {
                spyOn(category, 'executeOp');
                node.category = category.CATEGORY;
                operation_executor_1.executeOp(node, {}, context);
                expect(category.executeOp).toHaveBeenCalledWith(node, {}, context);
            });
        });
    });
});
//# sourceMappingURL=operation_executor_test.js.map