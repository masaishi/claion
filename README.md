# Clarion AI

Clarion AI is an innovative AI-powered tool designed to enhance global English communication by converting non-native English speech into clear, natural pronunciation in real-time.

## Overview

In today's globalized world, where English serves as a primary medium of communication for approximately 1.5 billion speakers, pronunciation barriers often hinder effective communication. Clarion AI addresses this challenge by providing real-time accent modification, enabling smoother international communication in various contexts such as:

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
│   ├── data/
│   │   ├── speechocean/    # Speech dataset handling
│   │   └── utils.py        # Utility functions
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

