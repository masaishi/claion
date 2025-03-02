# Claion

Claion is an innovative AI-powered tool designed to enhance global English communication by converting non-native English speech into clear, natural pronunciation in real-time.

The name "Claion" is derived from "Clarion," meaning "clear" or "distinct," with the "r" intentionally removed to include "AI," highlighting the tool's focus on enhancing speech clarity through AI.

## Overview

In today's globalized world, where English serves as a primary medium of communication for approximately 1.5 billion speakers, pronunciation barriers often hinder effective communication. Claion addresses this challenge by providing real-time accent modification, enabling smoother international communication in various contexts such as:

- Online meetings
- Remote interviews
- Educational content creation
- Personal pronunciation practice

## Technical Architecture

### Core Components

1. **SpeechT5 Integration**
   - Utilizes Microsoft's SpeechT5 for comprehensive voice processing
   - Converts input speech into semantic space representation
   - Regenerates speech with improved pronunciation while maintaining original content

2. **Accent Evaluation System**
   - Leverages OpenAI's Whisper model for accent assessment
   - Provides quantitative metrics for pronunciation improvement
   - Enables objective evaluation of voice conversion quality

3. **Reinforcement Learning Pipeline**
   - Implements continuous model improvement through reinforcement learning
   - Uses Whisper-based evaluation as reward signals
   - Optimizes for natural and clear pronunciation

## Project Structure

```
src/
├── clarion_ai/
│   ├── data/               # Data processing utilities
│   └── pipes/              # Processing pipelines
```

## Features

- Real-time voice conversion
- Accent modification while preserving speech content
- Low-latency processing
- High-quality audio output
- Integration with common communication platforms

## Technical Advantages

1. **Real-time Processing**
   - Minimal latency compared to traditional Speech-to-Text-to-Speech approaches
   - Optimized for live communication scenarios

2. **Quality Preservation**
   - Maintains original speech characteristics while improving pronunciation
   - Avoids audio quality degradation common in traditional conversion methods

3. **Continuous Improvement**
   - Self-improving system through reinforcement learning
   - Adapts to various accent patterns and speaking styles

## Applications

- **Business Communication**: Enhance clarity in international meetings and presentations
- **Education**: Support language learners and teachers
- **Content Creation**: Improve accessibility of educational and informational content
- **Personal Development**: Aid in pronunciation practice and speaking confidence

## Development Status

This project is currently under active development. The core components include:
- [x] Accent evaluation model implementation
- [ ] SpeechT5 integration
- [ ] Reinforcement learning pipeline
- [ ] Model optimization

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

## Contributing

Contribution guidelines are not yet available. Please stay tuned for updates.

## Contact

For inquiries, please contact mwishiha@ucsc.edu.

## Note

The intention is not to promote an English-centric approach or to hurt non-native speakers in any way. The developer of this tool grew up in Japan and moved to the United States for university. Despite living in an English-speaking environment, they do not consider themselves fluent in English. Especially during phone calls, it was often difficult to make themselves understood, and accents made communication even more challenging. This experience inspired them to leverage AI and technology to help bridge the gap and make communication smoother for everyone, regardless of their native language.