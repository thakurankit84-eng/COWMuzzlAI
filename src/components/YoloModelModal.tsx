import React, { useState } from 'react';
import { X, ExternalLink, Cpu, Database, CheckCircle2, Shield, Layers, FileCode2, Download, Network, ArrowRight, Sparkles, Activity } from 'lucide-react';
import { YOLOv8ModelInfo } from '../types';

interface YoloModelModalProps {
  isOpen: boolean;
  onClose: () => void;
  modelInfo: YOLOv8ModelInfo | null;
}

export const YoloModelModal: React.FC<YoloModelModalProps> = ({
  isOpen,
  onClose,
  modelInfo,
}) => {
  const [activeTab, setActiveTab] = useState<'resnet50' | 'yolo' | 'pipeline'>('resnet50');

  if (!isOpen) return null;

  const yoloDriveUrl = modelInfo?.googleDriveUrl || "https://drive.google.com/file/d/1fRoOOv7zmERFV0iBJCnBHfqSek2_hc55/view?usp=drive_link";
  const resnetDriveUrl = "https://drive.google.com/file/d/1-4gWM39-AbCKFV166_i9deyBhu43xU8y/view?usp=drive_link";

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-md animate-in fade-in duration-200">
      <div
        className="backdrop-blur-2xl bg-[#0a110a]/95 text-white rounded-3xl border border-white/15 shadow-2xl w-full max-w-3xl max-h-[92vh] overflow-y-auto flex flex-col"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Modal Header */}
        <div className="flex items-center justify-between p-5 border-b border-white/10 sticky top-0 bg-[#0a110a]/90 backdrop-blur-md z-10">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-2xl backdrop-blur-md bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 flex items-center justify-center shadow-lg shadow-emerald-500/10">
              <Cpu className="w-5 h-5" />
            </div>
            <div>
              <h3 className="text-lg font-bold text-white flex items-center gap-2">
                Deep AI Neural Models &amp; Biometrics
                <span className="text-[10px] uppercase font-mono px-2 py-0.5 rounded-full bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
                  Dual-Stage
                </span>
              </h3>
              <p className="text-xs text-slate-400 font-mono">
                ResNet-50 Feature Extractor &amp; YOLOv8 Muzzle Detector
              </p>
            </div>
          </div>

          <button
            id="close-model-modal-btn"
            onClick={onClose}
            className="p-2 rounded-xl text-slate-400 hover:text-white hover:bg-white/[0.08] transition-colors cursor-pointer border border-white/10"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Tab Navigation */}
        <div className="px-5 pt-3 border-b border-white/10 flex items-center gap-2">
          <button
            onClick={() => setActiveTab('resnet50')}
            className={`px-4 py-2 text-xs font-bold rounded-t-xl transition-all flex items-center gap-1.5 border-b-2 cursor-pointer ${
              activeTab === 'resnet50'
                ? 'border-emerald-400 text-emerald-400 bg-emerald-500/10'
                : 'border-transparent text-slate-400 hover:text-slate-200'
            }`}
          >
            <Network className="w-3.5 h-3.5" />
            <span>ResNet-50 Feature Extractor</span>
            <span className="px-1.5 py-0.2 bg-emerald-500/20 text-[9px] rounded text-emerald-300 font-mono">NEW</span>
          </button>

          <button
            onClick={() => setActiveTab('yolo')}
            className={`px-4 py-2 text-xs font-bold rounded-t-xl transition-all flex items-center gap-1.5 border-b-2 cursor-pointer ${
              activeTab === 'yolo'
                ? 'border-emerald-400 text-emerald-400 bg-emerald-500/10'
                : 'border-transparent text-slate-400 hover:text-slate-200'
            }`}
          >
            <Layers className="w-3.5 h-3.5" />
            <span>YOLOv8 Muzzle Detector</span>
          </button>

          <button
            onClick={() => setActiveTab('pipeline')}
            className={`px-4 py-2 text-xs font-bold rounded-t-xl transition-all flex items-center gap-1.5 border-b-2 cursor-pointer ${
              activeTab === 'pipeline'
                ? 'border-emerald-400 text-emerald-400 bg-emerald-500/10'
                : 'border-transparent text-slate-400 hover:text-slate-200'
            }`}
          >
            <Activity className="w-3.5 h-3.5" />
            <span>Pipeline &amp; FAISS Integration</span>
          </button>
        </div>

        {/* Modal Body */}
        <div className="p-5 sm:p-6 space-y-5 text-slate-200 text-xs sm:text-sm">
          {/* TAB 1: RESNET-50 */}
          {activeTab === 'resnet50' && (
            <div className="space-y-4">
              {/* Status Banner */}
              <div className="backdrop-blur-md bg-emerald-500/10 rounded-2xl p-4 border border-emerald-500/30 flex items-start gap-3">
                <CheckCircle2 className="w-5 h-5 text-emerald-400 mt-0.5 shrink-0" />
                <div>
                  <span className="font-bold text-emerald-300 block text-sm font-mono">
                    ResNet-50 Feature Extractor Active
                  </span>
                  <p className="text-xs text-emerald-200/80 mt-0.5">
                    Trained Deep Residual Neural Network (50 layers) specialized in mapping dermatoglyphic bead and ridge morphology into 128-dimensional L2-normalized vector embeddings for FAISS cosine similarity searches.
                  </p>
                </div>
              </div>

              {/* Model Attributes Table */}
              <div className="backdrop-blur-md bg-white/[0.03] rounded-2xl border border-white/10 overflow-hidden">
                <div className="p-3.5 bg-white/[0.04] border-b border-white/10 font-bold text-xs text-slate-200 flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Network className="w-4 h-4 text-emerald-400" />
                    ResNet-50 Specifications &amp; Parameters
                  </div>
                  <span className="font-mono text-[11px] text-emerald-400">Drive ID: 1-4gWM39-AbCKFV166_i9deyBhu43xU8y</span>
                </div>
                <div className="divide-y divide-white/[0.08] text-xs">
                  <div className="p-3.5 grid grid-cols-3 gap-2">
                    <span className="font-semibold text-slate-400">Model Name:</span>
                    <span className="col-span-2 font-mono font-bold text-white">
                      ResNet-50 Bovine Muzzle Biometric Feature Extractor
                    </span>
                  </div>

                  <div className="p-3.5 grid grid-cols-3 gap-2">
                    <span className="font-semibold text-slate-400">Weights File:</span>
                    <span className="col-span-2 font-mono font-bold text-emerald-400">
                      resnet50_cattle_muzzle_extractor.pth (97.8 MB)
                    </span>
                  </div>

                  <div className="p-3.5 grid grid-cols-3 gap-2">
                    <span className="font-semibold text-slate-400">Google Drive Source:</span>
                    <div className="col-span-2 flex items-center gap-2">
                      <a
                        href={resnetDriveUrl}
                        target="_blank"
                        rel="noreferrer"
                        className="text-emerald-400 hover:text-emerald-300 font-bold font-mono underline inline-flex items-center gap-1"
                      >
                        <span>View / Download on Google Drive</span>
                        <ExternalLink className="w-3 h-3" />
                      </a>
                    </div>
                  </div>

                  <div className="p-3.5 grid grid-cols-3 gap-2">
                    <span className="font-semibold text-slate-400">Backbone Architecture:</span>
                    <span className="col-span-2 text-slate-300">
                      ResNet-50 with 4 Residual Stages [3, 4, 6, 3 Bottleneck blocks with $1\times1 \to 3\times3 \to 1\times1$ Convolutions and Identity Shortcuts]
                    </span>
                  </div>

                  <div className="p-3.5 grid grid-cols-3 gap-2">
                    <span className="font-semibold text-slate-400">Input Resolution:</span>
                    <span className="col-span-2 font-mono text-slate-300">
                      224 &times; 224 &times; 3 (Normalized RGB Muzzle Patch)
                    </span>
                  </div>

                  <div className="p-3.5 grid grid-cols-3 gap-2">
                    <span className="font-semibold text-slate-400">Output Latent Space:</span>
                    <span className="col-span-2 font-mono text-emerald-300">
                      128-Dimensional L2-Normalized Unit Hypersphere Embedding ($\|v\|_2 = 1.0$)
                    </span>
                  </div>

                  <div className="p-3.5 grid grid-cols-3 gap-2">
                    <span className="font-semibold text-slate-400">Layer Decomposition:</span>
                    <div className="col-span-2 space-y-1 font-mono text-[11px] text-slate-300">
                      <div>• <span className="text-white">Conv1</span>: 7&times;7 Conv, 64 channels, stride 2 + MaxPool 3&times;3</div>
                      <div>• <span className="text-white">Layer 1</span>: 3&times; Bottleneck blocks (64 &rarr; 64 &rarr; 256 channels)</div>
                      <div>• <span className="text-white">Layer 2</span>: 4&times; Bottleneck blocks (128 &rarr; 128 &rarr; 512 channels)</div>
                      <div>• <span className="text-white">Layer 3</span>: 6&times; Bottleneck blocks (256 &rarr; 256 &rarr; 1024 channels)</div>
                      <div>• <span className="text-white">Layer 4</span>: 3&times; Bottleneck blocks (512 &rarr; 512 &rarr; 2048 channels)</div>
                      <div>• <span className="text-white">Global Pooling &amp; Head</span>: AdaptiveAvgPool2d(1) &rarr; Linear(2048 &rarr; 128) &rarr; L2 Normalize</div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* TAB 2: YOLOV8 */}
          {activeTab === 'yolo' && (
            <div className="space-y-4">
              {/* Status Banner */}
              <div className="backdrop-blur-md bg-emerald-500/10 rounded-2xl p-4 border border-emerald-500/30 flex items-start gap-3">
                <CheckCircle2 className="w-5 h-5 text-emerald-400 mt-0.5 shrink-0" />
                <div>
                  <span className="font-bold text-emerald-300 block text-sm font-mono">
                    YOLOv8 Muzzle Detection Model Active
                  </span>
                  <p className="text-xs text-emerald-200/80 mt-0.5">
                    Custom trained PyTorch YOLOv8 detection model specialized in detecting cattle snout/muzzle landmarks and extracting bounding crops.
                  </p>
                </div>
              </div>

              {/* Model Attributes Table */}
              <div className="backdrop-blur-md bg-white/[0.03] rounded-2xl border border-white/10 overflow-hidden">
                <div className="p-3.5 bg-white/[0.04] border-b border-white/10 font-bold text-xs text-slate-200 flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Layers className="w-4 h-4 text-emerald-400" />
                    YOLOv8 Parameters
                  </div>
                  <span className="font-mono text-[11px] text-emerald-400">Drive ID: 1fRoOOv7zmERFV0iBJCnBHfqSek2_hc55</span>
                </div>
                <div className="divide-y divide-white/[0.08] text-xs">
                  <div className="p-3.5 grid grid-cols-3 gap-2">
                    <span className="font-semibold text-slate-400">Model Name:</span>
                    <span className="col-span-2 font-mono font-bold text-white">
                      {modelInfo?.name || 'YOLOv8 Cattle Muzzle Detector'}
                    </span>
                  </div>

                  <div className="p-3.5 grid grid-cols-3 gap-2">
                    <span className="font-semibold text-slate-400">Weights File:</span>
                    <span className="col-span-2 font-mono font-bold text-emerald-400">
                      best_muzzle_detection_model.pt ({modelInfo?.sizeFormatted || '6.22 MB'})
                    </span>
                  </div>

                  <div className="p-3.5 grid grid-cols-3 gap-2">
                    <span className="font-semibold text-slate-400">Google Drive Source:</span>
                    <div className="col-span-2 flex items-center gap-2">
                      <a
                        href={yoloDriveUrl}
                        target="_blank"
                        rel="noreferrer"
                        className="text-emerald-400 hover:text-emerald-300 font-bold font-mono underline inline-flex items-center gap-1"
                      >
                        <span>View on Google Drive</span>
                        <ExternalLink className="w-3 h-3" />
                      </a>
                    </div>
                  </div>

                  <div className="p-3.5 grid grid-cols-3 gap-2">
                    <span className="font-semibold text-slate-400">Architecture:</span>
                    <span className="col-span-2 text-slate-300">
                      YOLOv8 Nano (CSPDarknet with C2f feature modules &amp; Decoupled Anchor-Free Head)
                    </span>
                  </div>

                  <div className="p-3.5 grid grid-cols-3 gap-2">
                    <span className="font-semibold text-slate-400">Input Resolution:</span>
                    <span className="col-span-2 font-mono text-slate-300">
                      640 &times; 640 &times; 3 (RGB Standard)
                    </span>
                  </div>

                  <div className="p-3.5 grid grid-cols-3 gap-2">
                    <span className="font-semibold text-slate-400">Detection Classes:</span>
                    <span className="col-span-2 font-mono text-slate-300">
                      Class 0: <span className="bg-emerald-500/20 text-emerald-300 px-2 py-0.5 rounded-lg border border-emerald-500/30 font-bold">muzzle</span> (Cow Snout / Rhinarium)
                    </span>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* TAB 3: PIPELINE */}
          {activeTab === 'pipeline' && (
            <div className="space-y-4">
              <div className="backdrop-blur-md bg-white/[0.03] rounded-2xl p-4 border border-white/10 space-y-3">
                <h4 className="text-xs font-bold text-white uppercase tracking-wider font-mono flex items-center gap-2">
                  <Activity className="w-4 h-4 text-emerald-400" />
                  Dual-Stage Biometric Identification Architecture
                </h4>
                
                <div className="space-y-2.5 text-xs text-slate-300">
                  <div className="p-3 rounded-xl bg-black/40 border border-white/10 flex items-center gap-3">
                    <div className="w-7 h-7 rounded-lg bg-emerald-500/20 text-emerald-400 flex items-center justify-center font-mono font-bold shrink-0">
                      1
                    </div>
                    <div className="flex-1">
                      <span className="font-bold text-white">Stage 1: YOLOv8 Muzzle Detection</span>
                      <p className="text-[11px] text-slate-400">Detects bounding coordinates of bovine muzzle, isolates nostrils and dermatoglyphic rhinarium.</p>
                    </div>
                  </div>

                  <div className="flex justify-center text-slate-500">
                    <ArrowRight className="w-4 h-4 rotate-90" />
                  </div>

                  <div className="p-3 rounded-xl bg-black/40 border border-white/10 flex items-center gap-3">
                    <div className="w-7 h-7 rounded-lg bg-blue-500/20 text-blue-400 flex items-center justify-center font-mono font-bold shrink-0">
                      2
                    </div>
                    <div className="flex-1">
                      <span className="font-bold text-white">Stage 2: ResNet-50 Deep Feature Extraction</span>
                      <p className="text-[11px] text-slate-400">Passes normalized 224&times;224 muzzle crop through 50 residual bottleneck layers &amp; global pooling to produce a 128-D L2-normalized vector embedding.</p>
                    </div>
                  </div>

                  <div className="flex justify-center text-slate-500">
                    <ArrowRight className="w-4 h-4 rotate-90" />
                  </div>

                  <div className="p-3 rounded-xl bg-black/40 border border-white/10 flex items-center gap-3">
                    <div className="w-7 h-7 rounded-lg bg-purple-500/20 text-purple-400 flex items-center justify-center font-mono font-bold shrink-0">
                      3
                    </div>
                    <div className="flex-1">
                      <span className="font-bold text-white">Stage 3: FAISS Vector Search &amp; Cosine Similarity</span>
                      <p className="text-[11px] text-slate-400">Calculates Inner Product cosine metric Cosine(A, B) against the indexed herd registry to identify matching cattle in &lt;1 ms.</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Biological Background */}
          <div className="backdrop-blur-md bg-white/[0.03] rounded-2xl p-4.5 border border-white/[0.08] space-y-2">
            <h4 className="text-xs font-bold text-white uppercase tracking-wider flex items-center gap-1.5 font-mono">
              <Shield className="w-4 h-4 text-teal-400" />
              Why Cattle Muzzle Biometrics?
            </h4>
            <p className="text-xs text-slate-300 leading-relaxed">
              Much like human fingerprints, the bead and groove patterns on a cow's muzzle (nasolabial dermatoglyphics) are 100% unique to each animal and remain invariant across its lifetime. Using YOLOv8 for localization and ResNet-50 for high-dimensional feature embeddings enables non-invasive, tamper-proof livestock identification, cattle insurance verification, and pedigree tracking without stressful ear tagging or branding.
            </p>
          </div>
        </div>

        {/* Modal Footer */}
        <div className="p-4.5 border-t border-white/10 flex flex-wrap items-center justify-between gap-2.5 bg-black/40 rounded-b-3xl">
          <div className="flex flex-wrap items-center gap-2">
            <a
              href="/api/download-model?type=resnet50"
              className="px-3.5 py-2.5 bg-emerald-500 hover:bg-emerald-400 text-black text-xs font-extrabold rounded-xl flex items-center gap-1.5 transition-all cursor-pointer shadow-lg shadow-emerald-500/20"
            >
              <Download className="w-3.5 h-3.5" />
              Download ResNet-50 (.pth)
            </a>

            <a
              href="/api/download-model?type=yolo"
              className="px-3.5 py-2.5 backdrop-blur-md bg-white/[0.08] hover:bg-white/[0.14] text-slate-200 text-xs font-semibold rounded-xl flex items-center gap-1.5 border border-white/10 transition-all cursor-pointer"
            >
              <Download className="w-3.5 h-3.5 text-emerald-400" />
              Download YOLOv8 (.pt)
            </a>

            <a
              href={activeTab === 'resnet50' ? resnetDriveUrl : yoloDriveUrl}
              target="_blank"
              rel="noreferrer"
              className="px-3.5 py-2.5 backdrop-blur-md bg-white/[0.08] hover:bg-white/[0.14] text-slate-200 text-xs font-semibold rounded-xl flex items-center gap-1.5 border border-white/10 transition-all cursor-pointer"
            >
              <ExternalLink className="w-3.5 h-3.5 text-emerald-400" />
              Drive Link
            </a>
          </div>

          <button
            onClick={onClose}
            className="px-4 py-2.5 backdrop-blur-md bg-white/[0.06] hover:bg-white/[0.12] text-slate-200 border border-white/10 text-xs font-bold rounded-xl transition-all cursor-pointer"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};

