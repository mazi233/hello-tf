use async_trait::async_trait;

use hello_tf::{
    infer_client::InferClient,
    process_client::ProcessClient,
    web_server::{Web, WebServer},
    InferRequest, InferResponse, PostProcessRequest, PostProcessResponse, PreProcessRequest,
    PreProcessResponse, Pred, WebRequest, WebResponse,
};
use tonic::{
    transport::{Channel, Server},
    Request, Response, Status,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let addr = "0.0.0.0:3001".parse()?;
    let web = WebImpl {
        infer_cli: InferClient::connect("http://localhost:5000").await?,
        process_cli: ProcessClient::connect("http://localhost:5001").await?,
    };
    Server::builder()
        .add_service(WebServer::new(web))
        .serve(addr)
        .await?;

    Ok(())
}

struct WebImpl {
    infer_cli: InferClient<Channel>,
    process_cli: ProcessClient<Channel>,
}

#[async_trait]
impl Web for WebImpl {
    async fn process(&self, req: Request<WebRequest>) -> Result<Response<WebResponse>, Status> {
        let mut infer_cli = self.infer_cli.clone();
        let mut process_cli = self.process_cli.clone();

        let PreProcessResponse { shape, data } =
            pre_process(&mut process_cli, &req.into_inner().image).await;
        let InferResponse { shape, data } = infer(&mut infer_cli, shape, data).await;
        let PostProcessResponse { preds } = post_process(&mut process_cli, shape, data).await;
        let preds: Vec<_> = preds
            .into_iter()
            .map(|p| Pred {
                name: p.name,
                probability: p.probability,
            })
            .collect();
        Ok(Response::new(WebResponse { preds }))
    }
}

async fn pre_process(cli: &mut ProcessClient<Channel>, data: &[u8]) -> PreProcessResponse {
    let req = PreProcessRequest { image: data.into() };
    cli.pre_process(req).await.unwrap().into_inner()
}

async fn infer(cli: &mut InferClient<Channel>, shape: Vec<u64>, data: Vec<f32>) -> InferResponse {
    let req = InferRequest { shape, data };
    cli.infer(req).await.unwrap().into_inner()
}

async fn post_process(
    cli: &mut ProcessClient<Channel>,
    shape: Vec<u64>,
    data: Vec<f32>,
) -> PostProcessResponse {
    let req = PostProcessRequest { shape, data };
    cli.post_process(req).await.unwrap().into_inner()
}
