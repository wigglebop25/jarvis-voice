use anyhow::*;
use flate2::read::GzDecoder;
use futures_util::StreamExt;
use reqwest::Client;
use std::fs::{File, remove_dir_all, remove_file};
use std::io::Write;
use std::path::{Path, PathBuf};
use tar::Archive;
use transcribe_rs::TranscriptionEngine;
use transcribe_rs::engines::parakeet::{ParakeetEngine, ParakeetModelParams};

pub async fn load_model(uri: &str, path: &str) -> Result<ParakeetEngine> {
    ensure_model_exists(uri, path).await?;

    let mut engine = ParakeetEngine::new();

    engine
        .load_model_with_params(path.as_ref(), ParakeetModelParams::int8())
        .map_err(|e| anyhow!(e.to_string()))
        .context("Failed to load model")?;

    println!("Transcription model loaded");

    Ok(engine)
}

async fn ensure_model_exists(uri: &str, path: &str) -> Result<()> {
    let archive_name = uri
        .split('/')
        .last()
        .ok_or_else(|| anyhow!("Invalid URI"))?;
    let archive_path = PathBuf::from(archive_name);

    if !Path::new(path).exists() {
        download_model(uri, &archive_path).await?;
        extract_archive(&archive_path).await?;
    } else {
        if !try_load_model(path.into()).await {
            remove_dir_all(path)?;
            download_model(uri, &archive_path).await?;
            extract_archive(&archive_path)
                .await
                .context("Failed to extract archive")?;
        }
    }

    Ok(())
}

async fn extract_archive(source: &PathBuf) -> Result<()> {
    let tar_gz = File::open(source)?;
    let tar = GzDecoder::new(tar_gz);
    let mut archive = Archive::new(tar);

    archive.unpack(".")?;

    remove_file(source)?;

    Ok(())
}

async fn try_load_model(path: PathBuf) -> bool {
    let mut engine = ParakeetEngine::new();

    let res = engine.load_model_with_params(&path, ParakeetModelParams::int8());

    res.is_ok()
}

async fn download_model(uri: &str, path: &PathBuf) -> Result<()> {
    let client = Client::new();
    let res = client
        .get(uri)
        .send()
        .await
        .context("failed to send request to client")?;

    println!("Model not found, downloading...");
    if !res.status().is_success() {
        bail!("failed to download model: {}", res.status());
    }

    println!("Model downloaded, extracting...");

    let mut file = File::create(path).context("failed to create file")?;

    let mut stream = res.bytes_stream();

    while let Some(item) = stream.next().await {
        let chunk = item.map_err(|e| anyhow!("Error while downloading chunk: {}", e))?;
        file.write_all(&chunk).context("Failed to write chunk")?;
    }

    println!("Model extracted");

    Ok(())
}
