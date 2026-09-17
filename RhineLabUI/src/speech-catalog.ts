/** One physical archive per callable ASR feature. IDs remain stable. */
export type SpeechFeature = 'live' | 'record' | 'target' | 'meeting' | 'voice';
const entries: { id: string; title: string; en: string; category: string; feature: SpeechFeature; model: string; abstract: string; findings: string[] }[] = [
  { id: 'X-001', title: '非实时识别 · SenseVoice', en: 'SENSEVOICE SMALL', category: '非实时普通识别', feature: 'record', model: 'sensevoice-small', abstract: '采集完成后整段转写。适合中英等多语言的快速识别，自动清理情感标签。', findings: ['使用 SenseVoice Small / GPU。', '支持麦克风、电脑音频和 WAV 文件。'] },
  { id: 'X-002', title: '非实时识别 · Fun-ASR-Nano', en: 'FUN-ASR-NANO', category: '非实时普通识别', feature: 'record', model: 'fun-asr-nano', abstract: '停止录音后，使用 Fun-ASR-Nano 生成整段转写；也可直接处理 WAV 文件。', findings: ['使用 PyTorch FP32 / GPU。', '本档案为非实时识别；准实时版本位于 X-006。'] },
  { id: 'X-003', title: '非实时识别 · Qwen3-ASR', en: 'QWEN3-ASR 1.7B', category: '非实时普通识别', feature: 'record', model: 'qwen3-asr', abstract: '整段语音识别，自动检测语言。首次加载和推理的显存占用较高。', findings: ['使用 Qwen3-ASR 1.7B / GPU。', '采集音频后停止，或导入 WAV，再生成文本。'] },
  { id: 'X-005', title: '实时识别 · SenseVoice', en: 'SENSEVOICE', category: '实时普通识别', feature: 'live', model: 'sensevoice-realtime', abstract: '分块重识别并实时更新文字。使用 CPU，预览允许修订，不是原生流式模型。', findings: ['使用 ONNX INT8 / CPU。', '语音结束后生成定稿。'] },
  { id: 'X-006', title: '实时识别 · Fun-ASR-Nano', en: 'FUN-ASR-NANO', category: '实时普通识别', feature: 'live', model: 'fun-asr-nano', abstract: 'FSMN-VAD 分段的准实时识别，持续更新预览，句尾定稿。', findings: ['使用 PyTorch FP32 / GPU。', 'WAV 按实时节奏输入，可停止并定稿；非实时版本位于 X-002。'] },
  { id: 'X-007', title: '目标说话人 · SenseVoice', en: 'TARGET SPEAKER', category: '目标说话人', feature: 'target', model: 'sensevoice-realtime', abstract: '仅转写与本机已注册声纹匹配的声音。FSMN-VAD + CAM++ + SenseVoice，全 CPU。', findings: ['先加载识别引擎和声纹组件；注册和删除声纹在 X-009。', '短语音可能跳过，不分离重叠声音；相似度不是概率。'] },
  { id: 'X-008', title: '会议转写 · MOSS', en: 'MOSS TRANSCRIBE', category: '多人会议', feature: 'meeting', model: 'moss-transcribe-diarize', abstract: '整段联合生成转写文字、说话人编号和时间戳。完成后可导出 JSON 与 SRT。', findings: ['使用 MOSS-Transcribe-Diarize 0.9B / GPU。说话人编号不代表真实身份。', '可取消会议推理；取消后需重新加载引擎。重叠讲话不保证识别无误。'] },
  { id: 'X-009', title: '声纹注册与管理', en: 'VOICE ENROLLMENT', category: '目标说话人', feature: 'voice', model: '', abstract: '通过麦克风、电脑音频或 WAV 注册 3–30 秒单人语音。声纹只保存在本机。', findings: ['重新注册将替换已有声纹，需要确认。', '注册仅提取声纹，不生成转写文本；不要混入其他人的声音。'] },
  { id: 'X-010', title: '实时识别 · 样品-X', en: 'SAMPLE-X / ORIGINAL A', category: '实时普通识别', feature: 'live', model: 'sample-x', abstract: '样品-X Sample-X 本地模型 + A 原前端流程。CUDA 优先、CPU 回退；浏览器采集、自动断句、草稿与定稿。', findings: ['CUDA FP32 解码 + MNN CPU 编码，Sample-X_v3.2.1；实际后端见功能面板。', '保留完整长段及 CPU 回退，并非原厂流式解码；支持麦克风、电脑音频与 WAV。'] },
];
export const speechColumns = ['实时普通识别', '目标说话人', '非实时普通识别', '多人会议'];
export const speechArchives = entries.map(item => ({
  ...item, department: item.model || 'FSMN-VAD + CAM++', date: '本地运行', lead: item.model === 'sample-x' ? 'AUTO' : item.feature === 'voice' || item.model === 'sensevoice-realtime' ? 'CPU' : 'GPU',
  clearance: item.model === 'sample-x' ? 'LOCAL / AUTO' : item.feature === 'voice' || item.model === 'sensevoice-realtime' ? 'LOCAL / CPU' : 'LOCAL / GPU',
  source: 'http://127.0.0.1:8765/rhine/',
}));
