#include "core/realtime_turn_buffer.h"

#include <catch2/catch_test_macros.hpp>
#include <vector>

TEST_CASE("realtime turns buffer without requesting prefix reprocessing", "[unit][realtime]") {
    core_realtime::TurnBuffer turn(10);
    const std::vector<float> first(4, 0.25f);
    const std::vector<float> second(5, -0.25f);
    const std::vector<float> last(1, 0.5f);

    REQUIRE_FALSE(turn.append(first.data(), first.size()).full);
    REQUIRE_FALSE(turn.append(second.data(), second.size()).full);
    REQUIRE(turn.size() == 9);
    REQUIRE(turn.append(last.data(), last.size()).full);
    REQUIRE(turn.size() == 10);
}

TEST_CASE("commit reset prevents audio leaking into the next turn", "[unit][realtime]") {
    core_realtime::TurnBuffer turn(10);
    const std::vector<float> audio(6, 0.25f);
    REQUIRE_FALSE(turn.append(audio.data(), audio.size()).full);
    turn.clear();
    REQUIRE(turn.empty());
    REQUIRE_FALSE(turn.append(audio.data(), 2).full);
    REQUIRE(turn.size() == 2);
}

TEST_CASE("a large append cannot exceed the turn safety cap", "[unit][realtime]") {
    core_realtime::TurnBuffer turn(10);
    const std::vector<float> audio(25, 0.25f);
    const auto appended = turn.append(audio.data(), audio.size());
    REQUIRE(appended.full);
    REQUIRE(appended.consumed == 10);
    REQUIRE(turn.size() == 10);
}

TEST_CASE("a caller can carry overflow into the next turn without loss", "[unit][realtime]") {
    core_realtime::TurnBuffer turn(10);
    const std::vector<float> audio(25, 0.25f);
    std::size_t offset = 0;
    int completed_turns = 0;
    while (offset < audio.size()) {
        const auto appended = turn.append(audio.data() + offset, audio.size() - offset);
        offset += appended.consumed;
        if (appended.full) {
            REQUIRE(turn.size() == 10);
            completed_turns++;
            turn.clear();
        }
    }
    REQUIRE(offset == audio.size());
    REQUIRE(completed_turns == 2);
    REQUIRE(turn.size() == 5);
}
