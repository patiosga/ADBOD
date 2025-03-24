from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.functions import ProcessFunction, RuntimeContext
from pyflink.common.typeinfo import Types
from pyflink.datastream.state import ValueStateDescriptor
from pyflink.common.time import Time
from pyflink.datastream.connectors import FlinkKafkaConsumer
from pyflink.common.serialization import SimpleStringSchema

import json
from datetime import datetime, timedelta    
import joblib
import numpy as np

from New_Dyn import optimized_dynamic



env = StreamExecutionEnvironment.get_execution_environment()
env.set_parallelism(1)


kafka_source = FlinkKafkaConsumer(
    topics='test',
    deserialization_schema=SimpleStringSchema(),
    properties={
        'bootstrap.servers': 'localhost:9092', 
        'group.id': 'test'
        }
    )

transaction_stream = env.add_source(kafka_source)

model = optimized_dynamic.dynamic_kr(slide=100, window=200, policy="or")


class EnhancedAnomalyDetection(ProcessFunction):
    def __init__(self):
        pass

    def open(self, runtime_context: RuntimeContext):
        descriptor = ValueStateDescriptor('last_transaction', Types.PICKLED_BYTE_ARRAY())
        self.last_transaction_state = runtime_context.get_state(descriptor)

    def process_timeseries(self, timeseries, ctx: 'ProcessFunction.Context'):
        transaction = json.loads(timeseries)
        last_transaction = self.last_transaction_state.value()

        if last_transaction is not None:
            last_transaction = json.loads(last_transaction)
            features = self.extract_features(last_transaction, transaction)

            scores = model.fit(features)


        self.last_transaction_state.update(json.dumps(transaction).encode())
            

