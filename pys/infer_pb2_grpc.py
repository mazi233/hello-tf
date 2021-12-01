# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import infer_pb2 as infer__pb2


class InferStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.Infer = channel.unary_unary(
                '/infer.Infer/Infer',
                request_serializer=infer__pb2.InferRequest.SerializeToString,
                response_deserializer=infer__pb2.InferResponse.FromString,
                )


class InferServicer(object):
    """Missing associated documentation comment in .proto file."""

    def Infer(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_InferServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'Infer': grpc.unary_unary_rpc_method_handler(
                    servicer.Infer,
                    request_deserializer=infer__pb2.InferRequest.FromString,
                    response_serializer=infer__pb2.InferResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'infer.Infer', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class Infer(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def Infer(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/infer.Infer/Infer',
            infer__pb2.InferRequest.SerializeToString,
            infer__pb2.InferResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)