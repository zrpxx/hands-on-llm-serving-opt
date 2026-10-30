# LLM Serving 实战教程

<div class="home-hero">
  <div>
    <h1 class="home-title">LLM Serving<br />实战教程</h1>
    <p class="home-lede">以《Hands-On LLM Serving and Optimization》为主线，从一次前向计算讲到一个能上线的推理服务：先弄清模型怎样生成、显存和瓶颈怎样估算，再自己实现连续批处理与分页 KV Cache，最后对照 2026 年的生产系统。</p>
    <div class="home-actions">
      <a class="is-primary" data-resume href="#/stage0/01-serving-landscape">从 0.1 开始</a>
      <a href="#/stage1/project1-walkthrough">项目 1 分步指南</a>
    </div>
  </div>
  <div class="llm-widget" data-widget="kv-pool"></div>
</div>

<h2 class="home-section-title">怎么学</h2>

<div class="home-facts">
  <div>
    <h4>先推导，再看公式</h4>
    <p>每个公式都用 Qwen3-8B、A100 这样的具体数字算一遍。原书中有误的地方，在相应位置单独注明。</p>
  </div>
  <div>
    <h4>用组件验证直觉</h4>
    <p>正文中的组件可以调参数或逐步执行：显存账本、Roofline、批处理时间线、分页块分配、前缀缓存命中。</p>
  </div>
  <div>
    <h4>每节一组自测</h4>
    <p>答完自测再点页底的"标记本节已完成"，侧边栏和下面的课程地图会记录进度。进度只保存在本机浏览器。</p>
  </div>
  <div>
    <h4>用项目收尾</h4>
    <p>每部分以一个项目结束。项目 1 在纯 CPU 上就能完成开发，每一步都有单独的测试。</p>
  </div>
</div>

<h2 class="home-section-title">课程地图</h2>
<p class="home-section-note">第 0～6 部分已经写完；其余部分目前是提纲，学习时边读书边补充。</p>

<div class="llm-widget" data-widget="course-map"></div>

<h2 class="home-section-title">前置要求</h2>

会写 Python 类和函数，知道张量的形状和矩阵乘法即可。异步编程、GPU 体系结构等内容在用到时讲解。第 0~1 部分的组件和项目 1 的开发都不需要 GPU；性能实验需要一张显卡，24 GB 起步，80 GB 更方便。

书中的 "Triton" 多数指 NVIDIA 的 Triton Inference Server（模型服务框架），第 3 部分用来写算子的是 OpenAI 的 Triton 语言，两者无关。标注"书外"的章节不在书中，需要额外读论文和源码。

<h2 class="home-section-title">在本地打开</h2>

```bash
git clone https://github.com/zrpxx/hands-on-llm-serving-opt.git
cd hands-on-llm-serving-opt
python3 -m http.server 3000      # 或 npx docsify-cli serve .
```

然后访问 `http://localhost:3000`。站点的全部依赖都在仓库里，不需要访问外部 CDN。
