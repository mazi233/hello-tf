use tensorflow::Graph;
use tensorflow::SavedModelBundle;
use tensorflow::SessionOptions;
use tensorflow::SessionRunArgs;
use tensorflow::Tensor;

use async_trait::async_trait;
use tokio::runtime::Builder;
use tonic::{transport::Server, Request, Response, Status};

use hello_tf::infer_server;
use hello_tf::InferRequest;
use hello_tf::InferResponse;

struct InferImpl {
    model: SavedModelBundle,
    graph: Graph,
}

impl InferImpl {
    fn new(export_dir: String) -> Self {
        let mut graph = Graph::new();
        let opts = SessionOptions::new();
        let model = SavedModelBundle::load(&opts, &["serve"], &mut graph, export_dir).unwrap();
        InferImpl { model, graph }
    }

    fn infer_impl(&self, request: Tensor<f32>) -> Tensor<f32> {
        let mut step = SessionRunArgs::new();

        let output_op = self
            .graph
            .operation_by_name_required("StatefulPartitionedCall")
            .unwrap();
        step.add_target(&output_op);

        let input = self
            .graph
            .operation_by_name_required("serving_default_input_1")
            .unwrap();
        step.add_feed(&input, 0, &request);

        let output_t = step.request_fetch(&output_op, 0);

        self.model.session.run(&mut step).unwrap();

        // 取出推理结果
        step.fetch(output_t).unwrap()
    }
}

#[async_trait]
impl infer_server::Infer for InferImpl {
    async fn infer(&self, req: Request<InferRequest>) -> Result<Response<InferResponse>, Status> {
        let req = req.into_inner();
        let request = Tensor::new(&req.shape).with_values(&req.data).unwrap();
        let output = self.infer_impl(request);
        let response = InferResponse {
            shape: output.dims().into(),
            data: output.to_vec(),
        };
        Ok(Response::new(response))
    }
}

fn main() {
    let rt = Builder::new_current_thread().enable_all().build().unwrap();
    rt.block_on(async {
        let addr = "0.0.0.0:5000";
        println!("Listen on: {}", addr);

        let addr = addr.parse().unwrap();
        let infer = InferImpl::new(String::from("pys/resnet50"));
        let server = infer_server::InferServer::new(infer);

        Server::builder()
            .add_service(server)
            .serve(addr)
            .await
            .unwrap();
    })
}
