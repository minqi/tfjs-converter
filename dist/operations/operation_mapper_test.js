"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var index_1 = require("../data/index");
var index_2 = require("./index");
var mapper = index_2.OperationMapper.Instance;
var graph;
var SIMPLE_MODEL = {
    node: [
        {
            name: 'image_placeholder',
            op: 'Placeholder',
            attr: {
                dtype: {
                    type: index_1.tensorflow.DataType.DT_FLOAT,
                },
                shape: { shape: { dim: [{ size: 3 }, { size: 3 }, { size: 3 }, { size: 1 }] } }
            }
        },
        {
            name: 'Const',
            op: 'Const',
            attr: {
                dtype: { type: index_1.tensorflow.DataType.DT_INT32 },
                value: {
                    tensor: {
                        dtype: index_1.tensorflow.DataType.DT_INT32,
                        tensorShape: { dim: [{ size: 3 }, { size: 3 }, { size: 1 }, { size: 1 }] },
                        intVal: [0, 0, 0, 0, 1, 0, 0, 0, 0]
                    }
                }
            }
        },
        {
            name: 'Shape',
            op: 'Const',
            attr: {
                dtype: { type: index_1.tensorflow.DataType.DT_INT32 },
                value: {
                    tensor: {
                        dtype: index_1.tensorflow.DataType.DT_INT32,
                        tensorShape: { dim: [{ size: 3 }, { size: 1 }, { size: 1 }, { size: 1 }] },
                        intVal: [1, 1, 1]
                    }
                }
            }
        },
        {
            name: 'Value',
            op: 'Const',
            attr: { dtype: { type: index_1.tensorflow.DataType.DT_INT32 }, value: { i: 1 } }
        },
        { name: 'Fill', op: 'Fill', input: ['Shape', 'Value'], attr: {} }, {
            name: 'Conv2D',
            op: 'Conv2D',
            input: ['image_placeholder', 'Const'],
            attr: {
                T: { type: index_1.tensorflow.DataType.DT_FLOAT },
                dataFormat: { s: Uint8Array.from([1, 12, 2]) },
                padding: { s: Uint8Array.from([118, 97, 108, 105, 100]) },
                strides: { list: { f: [], i: [1, 2, 2, 1] } },
                useCudnnOnGpu: { b: true }
            }
        },
        {
            name: 'BiasAdd',
            op: 'BiasAdd',
            input: ['Conv2D', 'Shape'],
            attr: {
                T: { type: index_1.tensorflow.DataType.DT_FLOAT },
                dataFormat: { s: Uint8Array.from([1, 2, 34]) }
            }
        }
    ],
    versions: { producer: 1.0 }
};
describe('operationMapper', function () {
    beforeEach(function () {
        graph = mapper.transformGraph(SIMPLE_MODEL);
    });
    afterEach(function () { });
    describe('transform graph', function () {
        describe('graph level', function () {
            it('should find the graph input nodes', function () {
                expect(graph.inputs.map(function (node) { return node.name; })).toEqual([
                    'image_placeholder', 'Const', 'Shape', 'Value'
                ]);
            });
            it('should find the graph output nodes', function () {
                expect(graph.outputs.map(function (node) { return node.name; })).toEqual([
                    'Fill', 'BiasAdd'
                ]);
            });
            it('should convert nodes', function () {
                expect(Object.keys(graph.nodes)).toEqual([
                    'image_placeholder', 'Const', 'Shape', 'Value', 'Fill', 'Conv2D',
                    'BiasAdd'
                ]);
            });
        });
        describe('node level', function () {
            it('should find the input nodes', function () {
                expect(graph.nodes['Fill'].inputs.map(function (node) { return node.name; })).toEqual([
                    'Shape', 'Value'
                ]);
            });
            it('should find the children nodes', function () {
                expect(graph.nodes['image_placeholder'].children.map(function (node) { return node.name; }))
                    .toEqual(['Conv2D']);
            });
            it('should map the input params', function () {
                expect(graph.nodes['Fill'].params['shape'].inputIndex).toEqual(0);
                expect(graph.nodes['Fill'].params['value'].inputIndex).toEqual(1);
            });
            it('should map the attribute params', function () {
                expect(graph.nodes['Conv2D'].params['strides'].value).toEqual([
                    1, 2, 2, 1
                ]);
                expect(graph.nodes['Conv2D'].params['pad'].value).toEqual('valid');
                expect(graph.nodes['Conv2D'].params['useCudnnOnGpu'].value)
                    .toEqual(true);
            });
        });
    });
});
//# sourceMappingURL=operation_mapper_test.js.map