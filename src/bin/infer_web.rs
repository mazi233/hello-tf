use axum::{
    extract::{Extension, Multipart},
    response::{Html, IntoResponse},
    routing::get,
    AddExtensionLayer, Router, http::StatusCode, Json,
};
use hello_tf::{
    infer_client::InferClient, process_client::ProcessClient, InferRequest, InferResponse,
    PostProcessRequest, PostProcessResponse, PreProcessRequest, PreProcessResponse,
};
use serde::Serialize;
use tokio::runtime::Builder;
use tonic::transport::Channel;

#[derive(Clone)]
struct Clients {
    infer_cli: InferClient<Channel>,
    process_cli: ProcessClient<Channel>,
}

fn main() {
    let rt = Builder::new_current_thread().enable_all().build().unwrap();

    rt.block_on(async {
        if std::env::var_os("RUST_LOG").is_none() {
            std::env::set_var("RUST_LOG", "infer_web=debug");
        }
        tracing_subscriber::fmt::init();

        let clients = Clients {
            infer_cli: InferClient::connect("http://localhost:5000").await.unwrap(),
            process_cli: ProcessClient::connect("http://localhost:5001")
                .await
                .unwrap(),
        };

        let app = Router::new()
            .route("/", get(show_form).post(classify_image))
            .layer(AddExtensionLayer::new(clients))
            .layer(tower_http::trace::TraceLayer::new_for_http());

        let addr = "0.0.0.0:3000".parse().unwrap();
        tracing::info!("Listening on {}", addr);
        axum::Server::bind(&addr)
            .serve(app.into_make_service())
            .await
            .unwrap();
    });
}

async fn show_form() -> Html<&'static str> {
    Html(include_str!("upload_form.html"))
}

async fn classify_image(
    mut multipart: Multipart,
    Extension(Clients {
        mut infer_cli,
        mut process_cli,
    }): Extension<Clients>,
) -> impl IntoResponse {
    let mut results = vec![];

    while let Some(field) = multipart.next_field().await.unwrap() {
        let image = field.file_name().unwrap().to_string();
        let data = field.bytes().await.unwrap();

        let PreProcessResponse { shape, data } = pre_process(&mut process_cli, &data).await;
        let InferResponse { shape, data } = infer(&mut infer_cli, shape, data).await;
        let PostProcessResponse { preds } = post_process(&mut process_cli, shape, data).await;
        let preds: Vec<_> = preds
            .into_iter()
            .map(|p| Pred {
                name: p.name,
                probability: p.probability,
            })
            .collect();
        results.push(Preds { image, preds })
    }
    (StatusCode::OK, Json(results))
}

#[derive(Serialize, Debug)]
struct Pred {
    name: String,
    probability: f32,
}

#[derive(Serialize, Debug)]
struct Preds {
    image: String,
    preds: Vec<Pred>,
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
