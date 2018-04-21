"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var ajv = require("ajv");
var schema = require("../op_mapper_schema.json");
var arithmetic = require("./arithmetic.json");
var basicMath = require("./basic_math.json");
var convolution = require("./convolution.json");
var creation = require("./creation.json");
var graph = require("./graph.json");
var image = require("./image.json");
var logical = require("./logical.json");
var matrices = require("./matrices.json");
var normalization = require("./normalization.json");
var reduction = require("./reduction.json");
var sliceJoin = require("./slice_join.json");
var transformation = require("./transformation.json");
describe('OpListTest', function () {
    var jsonValidator = new ajv();
    var validator = jsonValidator.compile(schema);
    beforeEach(function () { });
    describe('validate schema', function () {
        var mappersJson = {
            arithmetic: arithmetic,
            basicMath: basicMath,
            convolution: convolution,
            creation: creation,
            logical: logical,
            image: image,
            graph: graph,
            matrices: matrices,
            normalization: normalization,
            reduction: reduction,
            sliceJoin: sliceJoin,
            transformation: transformation
        };
        Object.keys(mappersJson).forEach(function (key) {
            it('should satisfy the schema: ' + key, function () {
                var valid = validator(mappersJson[key]);
                if (!valid)
                    console.log(validator.errors);
                expect(valid).toBeTruthy();
            });
        });
    });
});
//# sourceMappingURL=op_list_test.js.map